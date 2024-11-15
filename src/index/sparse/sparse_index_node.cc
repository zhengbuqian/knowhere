// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <sys/mman.h>

#include "index/sparse/sparse_inverted_index.h"
#include "index/sparse/sparse_inverted_index_config.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"
#include <iostream>

namespace knowhere {

// Inverted Index impl for sparse vectors. May optionally use WAND algorithm to speed up search.
//
// Not overriding RangeSearch, will use the default implementation in IndexNode.
//
// Thread safety: not thread safe.
template <typename T, bool use_wand>
class SparseInvertedIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, fp32>, "SparseInvertedIndexNode only support float");

 public:
    explicit SparseInvertedIndexNode(const int32_t& /*version*/, const Object& /*object*/)
        : search_pool_(ThreadPool::GetGlobalSearchThreadPool()), build_pool_(ThreadPool::GetGlobalBuildThreadPool()) {
    }

    ~SparseInvertedIndexNode() override {
        DeleteExistingIndex();
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);
        if (!IsMetricType(cfg.metric_type.value(), metric::IP) &&
            !IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            LOG_KNOWHERE_ERROR_ << Type() << " only support metric_type IP or BM25";
            return Status::invalid_metric_type;
        }
        auto drop_ratio_build = cfg.drop_ratio_build.value_or(0.0f);
        auto index_or = CreateIndex</*mmapped=*/false>(cfg);
        if (!index_or.has_value()) {
            return index_or.error();
        }
        auto index = index_or.value();
        index->Train(static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor()), dataset->GetRows(),
                     drop_ratio_build);
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, deleting old";
            DeleteExistingIndex();
        }
        index_ = index;
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> config) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not add data to empty " << Type();
            return Status::empty_index;
        }
        auto tryObj = build_pool_
                          ->push([&] {
                              return index_->Add(static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor()),
                                                 dataset->GetRows(), dataset->GetDim());
                          })
                          .getTry();
        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "failed to add data to index " << Type() << ": " << tryObj.exception().what();
            return Status::sparse_inner_error;
        }
        return tryObj.value();
    }

    [[nodiscard]] expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not search empty " << Type();
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);
        auto computer_or = index_->GetDocValueComputer(cfg);
        if (!computer_or.has_value()) {
            return expected<DataSetPtr>::Err(computer_or.error(), computer_or.what());
        }
        auto computer = computer_or.value();
        auto nq = dataset->GetRows();
        auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());
        auto k = cfg.k.value();
        auto refine_factor = cfg.refine_factor.value_or(10);
        auto drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f);

        auto p_id = std::make_unique<sparse::label_t[]>(nq * k);
        auto p_dist = std::make_unique<float[]>(nq * k);

        std::vector<folly::Future<folly::Unit>> futs;
        auto all_stats = std::make_unique<sparse::SearchStats[]>(nq);

        double search_count = 40;

        std::string user_input;
        LOG_KNOWHERE_WARNING_ << "Do you want to continue searching with new WAND? (less than 3 chars for yes): ";
        std::cin >> user_input;
        LOG_KNOWHERE_WARNING_ << "user_input: \"" << user_input << "\" decision: " << (user_input.size() < 3 ? "yes" : "no") << std::endl;
        if (user_input.size() < 3)
        {
            LOG_KNOWHERE_WARNING_ << "Start new WAND search: ";
            futs.reserve(nq * search_count);
            auto now = std::chrono::steady_clock::now();
            for (int64_t count = 0; count < nq * search_count; ++count) {
                int64_t idx = count % nq;
                futs.emplace_back(search_pool_->push([&, idx = idx, p_id = p_id.get(), p_dist = p_dist.get(), all_stats = all_stats.get()]() {
                    auto stats = index_->Search(queries[idx], k, drop_ratio_search, p_dist + idx * k, p_id + idx * k, refine_factor,
                                bitset, computer, true);
                    all_stats[idx] = stats;
                }));
            }
            WaitAllSuccess(futs);
            auto end = std::chrono::steady_clock::now();
            sparse::SearchStats total_stats;
            for (int64_t i = 0; i < nq; ++i) {
                total_stats.push_count += all_stats[i].push_count;
                total_stats.score_count += all_stats[i].score_count;
                total_stats.swap_count += all_stats[i].swap_count;
                total_stats.pivot_count += all_stats[i].pivot_count;
                total_stats.break_early_count += all_stats[i].break_early_count;
                total_stats.init_upper_bound_count += all_stats[i].init_upper_bound_count;
                total_stats.sort_cursor_count += all_stats[i].sort_cursor_count;
            }
            LOG_KNOWHERE_WARNING_ << "New WAND search stats:";
            if (total_stats.push_count > 0) {
                LOG_KNOWHERE_WARNING_ << "\t\tAverage push count: " << static_cast<double>(total_stats.push_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage score count: " << static_cast<double>(total_stats.score_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage swap count: " << static_cast<double>(total_stats.swap_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage pivot count: " << static_cast<double>(total_stats.pivot_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage break early count: " << static_cast<double>(total_stats.break_early_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage init upper bound count: " << static_cast<double>(total_stats.init_upper_bound_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage sort cursor count: " << static_cast<double>(total_stats.sort_cursor_count) / nq;
            }
            LOG_KNOWHERE_WARNING_ << "\t\tTime taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count() << "ms";
        }

        LOG_KNOWHERE_WARNING_ << "Do you want to continue searching with old WAND? (less than 3 chars for yes): ";
        std::cin >> user_input;
        LOG_KNOWHERE_WARNING_ << "user_input: \"" << user_input << "\" decision: " << (user_input.size() < 3 ? "yes" : "no") << std::endl;
        if (user_input.size() < 3)
        {
            LOG_KNOWHERE_WARNING_ << "Start old WAND search: ";
            futs.clear();
            all_stats = std::make_unique<sparse::SearchStats[]>(nq);
            futs.reserve(nq);
            auto now = std::chrono::steady_clock::now();
            for (int64_t count = 0; count < nq * search_count; ++count) {
                int64_t idx = count % nq;
                futs.emplace_back(search_pool_->push([&, idx = idx, p_id = p_id.get(), p_dist = p_dist.get(), all_stats = all_stats.get()]() {
                    auto stats = index_->Search(queries[idx], k, drop_ratio_search, p_dist + idx * k, p_id + idx * k, refine_factor,
                                bitset, computer, false);
                    all_stats[idx] = stats;
                }));
            }
            WaitAllSuccess(futs);
            auto end = std::chrono::steady_clock::now();
            sparse::SearchStats total_stats;
            total_stats.push_count = 0;
            total_stats.score_count = 0;
            total_stats.swap_count = 0;
            total_stats.pivot_count = 0;
            total_stats.break_early_count = 0;
            total_stats.init_upper_bound_count = 0;
            total_stats.sort_cursor_count = 0;

            for (int64_t i = 0; i < nq; ++i) {
                total_stats.push_count += all_stats[i].push_count;
                total_stats.score_count += all_stats[i].score_count;
                total_stats.swap_count += all_stats[i].swap_count;
                total_stats.pivot_count += all_stats[i].pivot_count;
                total_stats.break_early_count += all_stats[i].break_early_count;
                total_stats.init_upper_bound_count += all_stats[i].init_upper_bound_count;
                total_stats.sort_cursor_count += all_stats[i].sort_cursor_count;
            }
            LOG_KNOWHERE_WARNING_ << "Old WAND search stats:";
            if (total_stats.push_count > 0) {
                LOG_KNOWHERE_WARNING_ << "\t\tAverage push count: " << static_cast<double>(total_stats.push_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage score count: " << static_cast<double>(total_stats.score_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage swap count: " << static_cast<double>(total_stats.swap_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage pivot count: " << static_cast<double>(total_stats.pivot_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage break early count: " << static_cast<double>(total_stats.break_early_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage init upper bound count: " << static_cast<double>(total_stats.init_upper_bound_count) / nq;
                LOG_KNOWHERE_WARNING_ << "\t\tAverage sort cursor count: " << static_cast<double>(total_stats.sort_cursor_count) / nq;
            }
            LOG_KNOWHERE_WARNING_ << "\t\tTime taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count() << "ms";
        }


        auto ret_ds = std::make_shared<DataSet>();
        ret_ds->SetRows(nq);
        ret_ds->SetDim(k);
        ret_ds->SetIds((const int64_t*)p_id.release());
        ret_ds->SetDistance(p_dist.release());
        // ret_ds->SetLims((const size_t*)push_counts.release());
        ret_ds->SetIsOwner(true);
        return ret_ds;
    }

 private:
    class RefineIterator : public IndexIterator {
     public:
        RefineIterator(const sparse::BaseInvertedIndex<T>* index, sparse::SparseRow<T>&& query,
                       std::shared_ptr<PrecomputedDistanceIterator> precomputed_it,
                       const sparse::DocValueComputer<T>& computer, const float refine_ratio = 0.5f)
            : IndexIterator(true, refine_ratio),
              index_(index),
              query_(std::move(query)),
              computer_(computer),
              precomputed_it_(precomputed_it) {
        }

     protected:
        // returns n_rows / 10 DistId for the first time to create a large enough window for refinement.
        void
        next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) override {
            std::vector<DistId> dists;
            size_t num = first_return_ ? (std::max(index_->n_rows() / 10, static_cast<size_t>(20))) : 1;
            first_return_ = false;
            for (size_t i = 0; i < num && precomputed_it_->HasNext(); ++i) {
                auto [id, dist] = precomputed_it_->Next();
                dists.emplace_back(id, dist);
            }
            batch_handler(dists);
        }

        float
        raw_distance(int64_t id) override {
            return index_->GetRawDistance(id, query_, computer_);
        }

     private:
        const sparse::BaseInvertedIndex<T>* index_;
        sparse::SparseRow<T> query_;
        const sparse::DocValueComputer<T> computer_;
        std::shared_ptr<PrecomputedDistanceIterator> precomputed_it_;
        bool first_return_ = true;
    };

 public:
    // TODO: for now inverted index and wand use the same impl for AnnIterator.
    [[nodiscard]] expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "creating iterator on empty index";
            return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(Status::empty_index,
                                                                                    "index not loaded");
        }
        auto nq = dataset->GetRows();
        auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);
        auto computer_or = index_->GetDocValueComputer(cfg);
        if (!computer_or.has_value()) {
            return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(computer_or.error(),
                                                                                    computer_or.what());
        }
        auto computer = computer_or.value();
        auto drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f);

        const bool approximated = index_->IsApproximated() || drop_ratio_search > 0;

        auto vec = std::vector<std::shared_ptr<IndexNode::iterator>>(nq, nullptr);
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.emplace_back(search_pool_->push([&, i]() {
                auto it = std::make_shared<PrecomputedDistanceIterator>(
                    index_->GetAllDistances(queries[i], drop_ratio_search, bitset, computer), true);
                if (!approximated || queries[i].size() == 0) {
                    vec[i] = it;
                } else {
                    sparse::SparseRow<T> query_copy(queries[i]);
                    auto refine_it = std::make_shared<RefineIterator>(index_, std::move(query_copy), it, computer);
                    refine_it->initialize();
                    vec[i] = std::move(refine_it);
                }
            }));
        }
        WaitAllSuccess(futs);

        return vec;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();

        auto data = std::make_unique<sparse::SparseRow<T>[]>(rows);
        int64_t dim = 0;
        try {
            for (int64_t i = 0; i < rows; ++i) {
                auto& target = data[i];
                index_->GetVectorById(ids[i], target);
                dim = std::max(dim, target.dim());
            }
        } catch (std::exception& e) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed");
        }
        auto res = GenResultDataSet(rows, dim, data.release());
        res->SetIsSparse(true);
        return res;
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        throw std::runtime_error("GetIndexMeta not supported for current index type");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not serialize empty " << Type();
            return Status::empty_index;
        }
        MemoryIOWriter writer;
        RETURN_IF_ERROR(index_->Save(writer));
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, deleting old";
            DeleteExistingIndex();
        }
        auto cfg = static_cast<const knowhere::SparseInvertedIndexConfig&>(*config);
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid BinarySet.";
            return Status::invalid_binary_set;
        }
        MemoryIOReader reader(binary->data.get(), binary->size);
        auto index_or = CreateIndex</*mmapped=*/false>(cfg);
        if (!index_or.has_value()) {
            return index_or.error();
        }
        index_ = index_or.value();
        return index_->Load(reader);
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, deleting old";
            DeleteExistingIndex();
        }
        auto cfg = static_cast<const knowhere::SparseInvertedIndexConfig&>(*config);
        auto reader = knowhere::FileReader(filename);
        map_size_ = reader.size();
        int map_flags = MAP_SHARED;
#ifdef MAP_POPULATE
        if (cfg.enable_mmap_pop.has_value() && cfg.enable_mmap_pop.value()) {
            map_flags |= MAP_POPULATE;
        }
#endif
        map_ = static_cast<char*>(mmap(nullptr, map_size_, PROT_READ, map_flags, reader.descriptor(), 0));
        if (map_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to mmap file: " << strerror(errno);
            return Status::disk_file_error;
        }
        if (madvise(map_, map_size_, MADV_RANDOM) != 0) {
            LOG_KNOWHERE_WARNING_ << "Failed to madvise file: " << strerror(errno);
        }
        auto index_or = CreateIndex</*mmapped=*/true>(cfg);
        if (!index_or.has_value()) {
            return index_or.error();
        }
        index_ = index_or.value();
        MemoryIOReader map_reader((uint8_t*)map_, map_size_);
        auto supplement_target_filename = filename + ".knowhere_sparse_index_supplement";
        return index_->Load(map_reader, map_flags, supplement_target_filename);
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SparseInvertedIndexConfig>();
    }

    [[nodiscard]] std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    // note that the Dim of a sparse vector index may change as new vectors are added
    [[nodiscard]] int64_t
    Dim() const override {
        return index_ ? index_->n_cols() : 0;
    }

    [[nodiscard]] int64_t
    Size() const override {
        return index_ ? index_->size() : 0;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_ ? index_->n_rows() : 0;
    }

    [[nodiscard]] std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    }

 private:
    template <bool mmapped>
    expected<sparse::BaseInvertedIndex<T>*>
    CreateIndex(const SparseInvertedIndexConfig& cfg) const {
        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            auto idx = new sparse::InvertedIndex<T, use_wand, true, mmapped>();
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return expected<sparse::BaseInvertedIndex<T>*>::Err(
                    Status::invalid_args, "BM25 parameters k1, b, and avgdl must be set when building/loading");
            }
            auto k1 = cfg.bm25_k1.value();
            auto b = cfg.bm25_b.value();
            auto avgdl = cfg.bm25_avgdl.value();
            auto max_score_ratio = cfg.wand_bm25_max_score_ratio.value();
            idx->SetBM25Params(k1, b, avgdl, max_score_ratio);
            return idx;
        } else {
            return new sparse::InvertedIndex<T, use_wand, false, mmapped>();
        }
    }

    void
    DeleteExistingIndex() {
        if (index_ != nullptr) {
            delete index_;
            index_ = nullptr;
        }
        if (map_ != nullptr) {
            auto res = munmap(map_, map_size_);
            if (res != 0) {
                LOG_KNOWHERE_ERROR_ << "Failed to munmap when trying to delete index: " << strerror(errno);
            }
            map_ = nullptr;
            map_size_ = 0;
        }
    }

    sparse::BaseInvertedIndex<T>* index_{};
    std::shared_ptr<ThreadPool> search_pool_;
    std::shared_ptr<ThreadPool> build_pool_;

    // if map_ is not nullptr, it means the index is mmapped from disk.
    char* map_ = nullptr;
    size_t map_size_ = 0;
};  // class SparseInvertedIndexNode

// Concurrent version of SparseInvertedIndexNode
//
// Thread safety: only the overridden methods are allowed to be called concurrently.
template <typename T, bool use_wand>
class SparseInvertedIndexNodeCC : public SparseInvertedIndexNode<T, use_wand> {
 public:
    explicit SparseInvertedIndexNodeCC(const int32_t& version, const Object& object)
        : SparseInvertedIndexNode<T, use_wand>(version, object) {
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t task_id = next_task_id_++;
        add_tasks_.push(task_id);

        // add task is allowed to run only after all search tasks that come before it have finished.
        cv_.wait(lock, [this, task_id]() { return current_task_id_ == task_id && active_readers_ == 0; });

        auto res = SparseInvertedIndexNode<T, use_wand>::Add(dataset, cfg);

        add_tasks_.pop();
        current_task_id_++;
        lock.unlock();
        cv_.notify_all();
        return res;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Search(dataset, std::move(cfg), bitset);
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        ReadPermission permission(*this);
        // Always uses PrecomputedDistanceIterator for SparseInvertedIndexNodeCC:
        // If we want to use RefineIterator, it needs to get another ReadPermission when calling
        // index_->GetRawDistance(). If an Add task is added in between, there will be a deadlock.
        auto config = static_cast<const knowhere::SparseInvertedIndexConfig&>(*cfg);
        config.drop_ratio_search = 0.0f;
        return SparseInvertedIndexNode<T, use_wand>::AnnIterator(dataset, std::move(cfg), bitset);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::RangeSearch(dataset, std::move(cfg), bitset);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::GetVectorByIds(dataset);
    }

    int64_t
    Dim() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Dim();
    }

    int64_t
    Size() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Size();
    }

    int64_t
    Count() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Count();
    }

    std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND_CC
                        : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC;
    }

 private:
    struct ReadPermission {
        ReadPermission(const SparseInvertedIndexNodeCC& node) : node_(node) {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            uint64_t task_id = node_.next_task_id_++;
            // read task may execute only after all add tasks that come before it have finished.
            if (!node_.add_tasks_.empty() && task_id > node_.add_tasks_.front()) {
                node_.cv_.wait(
                    lock, [this, task_id]() { return node_.add_tasks_.empty() || task_id < node_.add_tasks_.front(); });
            }
            // read task is allowed to run, block all add tasks
            node_.active_readers_++;
        }

        ~ReadPermission() {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            node_.active_readers_--;
            node_.current_task_id_++;
            node_.cv_.notify_all();
        }
        const SparseInvertedIndexNodeCC& node_;
    };

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable int64_t active_readers_ = 0;
    mutable std::queue<uint64_t> add_tasks_;
    mutable uint64_t next_task_id_ = 0;
    mutable uint64_t current_task_id_ = 0;
};  // class SparseInvertedIndexNodeCC

KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX_CC, SparseInvertedIndexNodeCC,
                                             knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND_CC, SparseInvertedIndexNodeCC, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
}  // namespace knowhere
