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

#include <future>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"
#include "index/sparse/sparse_inverted_index.h"
#include "index/sparse/sparse_inverted_index_config.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "utils.h"
#include <iomanip>
#include <sstream>


knowhere::DataSetPtr
LoadSparseEmbeddings(const std::string& file_path, float& avgdl) {
    std::uintmax_t filesize = std::filesystem::file_size(file_path);
    char* buf = new char[filesize];
    std::ifstream fin(file_path, std::ios::binary);
    fin.read(buf, filesize);
    if(!fin) {
        std::cerr << "Error reading file, could only read " << fin.gcount() << " bytes" << std::endl;
    }
    fin.close();

    int64_t rows, cols, nnz;
    uint8_t* data = (uint8_t*)buf;
    rows = *(int64_t*)data;
    data += sizeof(int64_t);
    cols = *(int64_t*)data;
    data += sizeof(int64_t);
    nnz = *(int64_t*)data;
    data += sizeof(int64_t);

    int64_t* indptr = (int64_t*)data;
    data += (rows + 1) * sizeof(int64_t);
    int32_t* indices = (int32_t*)data;
    data += nnz * sizeof(int32_t);
    float* values = (float*)data;

    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(rows);

    double total_values = 0.0;

    for (int64_t i = 0; i < rows; i++) {
        auto cnt = indptr[i + 1] - indptr[i];
        if (cnt == 0) {
            continue;
        }
        knowhere::sparse::SparseRow<float> row(cnt);
        std::vector<knowhere::sparse::SparseIdVal<float>> vec(cnt);
        for (int64_t j = 0; j < cnt; j++) {
            vec[j].id = indices[indptr[i] + j];
            vec[j].val = values[indptr[i] + j];
        }
        // sort vec by id, smaller id first
        std::sort(vec.begin(), vec.end(), [](const knowhere::sparse::SparseIdVal<float>& a, const knowhere::sparse::SparseIdVal<float>& b) {
            return a.id < b.id;
        });
        for (int64_t j = 0; j < cnt; j++) {
            // if (i > 100000 && i < 100010) {
            //     LOG_KNOWHERE_INFO_ << "row " << i << " id: " << vec[j].id << ", val: " << vec[j].val << std::endl;
            // }
            row.set_at(j, vec[j].id, vec[j].val);
            total_values += vec[j].val;
            if (j > 0) {
                REQUIRE(vec[j].id > vec[j - 1].id);
            }
        }
        tensor[i] = std::move(row);
    }

    auto ttl_nnz = 0;
    for (auto i = 0; i < rows; i++) {
        ttl_nnz += tensor[i].size();
    }

    auto num_to_str = [](auto num) -> std::string {
        if (std::abs(num) >= 1000000) {
            double millions = num / 1000000.0;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << millions << "M";
            return ss.str();
        }

        if constexpr (std::is_integral_v<decltype(num)>) {
            return std::to_string(num);
        } else {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << num;
            return ss.str();
        }
    };

    avgdl = total_values / rows;

    LOG_KNOWHERE_INFO_ << "Loading data from file: " << std::filesystem::path(file_path).filename() << std::endl;
    LOG_KNOWHERE_INFO_ << "rows " << num_to_str(rows) << " cols " << num_to_str(cols) << " total nnz: " << num_to_str(ttl_nnz)
                       << " avg nnz: " << num_to_str(static_cast<float>(ttl_nnz) / rows)
                       << " sum of all values: " << num_to_str(total_values) << " avg values: " << num_to_str(total_values / ttl_nnz)
                       << " avgdl: " << num_to_str(avgdl) << std::endl;

    auto ds = knowhere::GenDataSet(rows, cols, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);

    delete[] buf;
    return ds;
}

TEST_CASE("Test Mem Sparse Index MsMarco BM25 index", "[float metrics]") {
    static const std::string PINECONE_DOCS_FILE = "/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_document_bm25.csr";
    static const std::string PINECONE_QUERIES_FILE = "/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_query_bm25.csr";
    static const std::string MILVUS_DOCS_FILE = "/home/zilliz/datasets/sparse-custom/ms_marco_milvus_bm25/ms_marco_passage_doc_tf_vectors_default_tokenizer.csr";
    static const std::string MILVUS_QUERIES_FILE = "/home/zilliz/datasets/sparse-custom/ms_marco_milvus_bm25/ms_marco_passage_query_idf_vectors_default_tokenizer.csr";
    static const std::string SPLADE_DOCS_FILE = "/home/zilliz/datasets/sparse-bigann-23/base_full.csr";
    static const std::string SPLADE_QUERIES_FILE = "/home/zilliz/datasets/sparse-bigann-23/queries.dev.csr";



    auto [d_file, q_file] = GENERATE(table<std::string, std::string>({
        {PINECONE_DOCS_FILE, PINECONE_QUERIES_FILE},
        // {MILVUS_DOCS_FILE, MILVUS_QUERIES_FILE},
        // {SPLADE_DOCS_FILE, SPLADE_QUERIES_FILE},
    }));

    float avgdl_docs;
    auto docs = LoadSparseEmbeddings(d_file, avgdl_docs);
    float avgdl_queries;
    auto queries = LoadSparseEmbeddings(q_file, avgdl_queries);

    std::vector<int> topks = {10};
    std::vector<float> drop_ratio_builds = {0.32};
    std::vector<float> drop_ratio_searchs = {0.6};
    std::vector<const char*> index_names = {
        // knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
        knowhere::IndexEnum::INDEX_SPARSE_WAND
        };
    std::vector<const char*> metrics = {knowhere::metric::BM25}; // knowhere::metric::IP
    auto version = GenTestVersionList();
    // auto nq = 6980;


    auto doc_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(docs->GetTensor());
    auto doc_rows = docs->GetRows();
    std::vector<float> row_sums(doc_rows, 0.0f);


    for (int i = 0; i < doc_rows; i++) {
        auto& doc = doc_tensor[i];
        for (int j = 0; j < doc.size(); j++) {
            auto [dim, value] = doc[j];
            row_sums[i] += value;
        }
    }
    // float avgdl = std::accumulate(row_sums.begin(), row_sums.end(), 0.0f) / doc_rows;
    // std::cout << "avgdl: " << avgdl << std::endl;
    // // 12.7782 is the actual avgdl of the docs
    float used_avgdl = avgdl_docs;
    // int refine_factor = GENERATE(1, 10);
    int refine_factor = 1;

    auto dim = std::max(docs->GetDim(), queries->GetDim());

    for (auto metric : metrics) {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
        LOG_KNOWHERE_INFO_ << "[metric " << metric << " avgdl " << used_avgdl << " refine_factor " << refine_factor << "]" << std::endl;// BruteForce search time: " << duration.count() << "ms" << std::endl;

        for (auto index_name : index_names) {
            LOG_KNOWHERE_INFO_ << "\t Index: " << index_name << std::endl;
            for (auto drop_ratio_build : drop_ratio_builds) {
                auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_name, version).value();

                knowhere::Json json;
                json[knowhere::meta::DIM] = dim;
                json[knowhere::meta::METRIC_TYPE] = metric;
                json[knowhere::meta::BM25_K1] = 1.2;
                json[knowhere::meta::BM25_B] = 0.75;
                json["refine_factor"] = refine_factor;
                json[knowhere::meta::BM25_AVGDL] = used_avgdl;
                json[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;

                REQUIRE(idx.Build(docs, json) == knowhere::Status::success);
                REQUIRE(idx.Size() > 0);
                REQUIRE(idx.Count() == docs->GetRows());
                REQUIRE(idx.Type() == index_name);

                for (auto topk : topks) {
                    for (auto drop_ratio_search : drop_ratio_searchs) {
                        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
                        json[knowhere::meta::TOPK] = topk;

                        now = std::chrono::system_clock::now();
                        LOG_KNOWHERE_INFO_ << "start search drop_ratio_search " << drop_ratio_search << " topk " << topk << std::endl;
                        // for (int i = 0; i < 100; i++) {
                            auto results = idx.Search(queries, json, nullptr);
                            REQUIRE(results.has_value());
                        // }
                        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
                        LOG_KNOWHERE_INFO_ << "\t\t[ " << drop_ratio_build << ", " << drop_ratio_search << ", " << used_avgdl << "] Index search time: " << duration.count() << "ms" << std::endl;
                    }
                }
            }
        }
        LOG_KNOWHERE_INFO_ << "\n" << std::endl;
    }

}

// Below 4 tests re-implement the sparse inverted index for verification.

// TEST_CASE("Test Mem Sparse Index MsMarco BM25 Compute BM25 basic IP", "[float metrics]") {
//     LOG_KNOWHERE_INFO_ << "start basic IP" << std::endl;
//     auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_document_bm25.csr");
//     auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_query_bm25.csr");

//     // auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_collection.csr");
//     // auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_query.csr");
//     // LOG_KNOWHERE_INFO_ << "docs rows: " << docs->GetRows() << ", cols: " << docs->GetDim() << std::endl;
//     // LOG_KNOWHERE_INFO_ << "queries rows: " << queries->GetRows() << ", cols: " << queries->GetDim() << std::endl;

//     auto doc_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(docs->GetTensor());
//     auto doc_rows = docs->GetRows();
//     auto query_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(queries->GetTensor());
//     auto query_rows = queries->GetRows();

//     // key: dim, value: list of vec ids and values
//     std::unordered_map<knowhere::sparse::table_t, std::vector<knowhere::sparse::SparseIdVal<float>>> inverted_lut;

//     for (int i = 0; i < doc_rows; i++) {
//         auto& doc = doc_tensor[i];
//         for (int j = 0; j < doc.size(); j++) {
//             auto [dim, value] = doc[j];
//             inverted_lut[dim].emplace_back(i, value);
//         }
//     }

//     std::vector<float> ttl_scores(query_rows, 0.0f);
//     auto now = std::chrono::system_clock::now();
//     std::vector<folly::Future<folly::Unit>> futs;
//     auto pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
//     futs.reserve(query_rows);

//     for (int i = 0; i < query_rows; i++) {
//         futs.emplace_back(pool->push([&, i]() {
//             std::vector<float> scores(doc_rows, 0.0f);
//             auto& query = query_tensor[i];
//             for (int j = 0; j < query.size(); j++) {
//                 auto [dim, value] = query[j];
//                 auto& ids_and_values = inverted_lut[dim];
//                 for (auto& [id, val] : ids_and_values) {
//                     scores[id] += val * value;
//                 }
//             }
//             for (int j = 0; j < doc_rows; j++) {
//                 ttl_scores[i] += scores[j];
//             }
//         }));
//     }
//     knowhere::WaitAllSuccess(futs);

//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
//     LOG_KNOWHERE_INFO_ << "compute IP time: " << duration.count() / 1000 << "s" << std::endl;
//     auto sum_scores = std::accumulate(ttl_scores.begin(), ttl_scores.end(), 0.0f);
//     LOG_KNOWHERE_INFO_ << "sum scores: " << sum_scores << std::endl << std::endl << std::endl;

// }

// TEST_CASE("Test Mem Sparse Index MsMarco BM25 Compute BM25 basic", "[float metrics]") {
//     LOG_KNOWHERE_INFO_ << "start basic" << std::endl;
//     auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_document_bm25.csr");
//     auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_query_bm25.csr");

//     // auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_collection.csr");
//     // auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_query.csr");
//     // LOG_KNOWHERE_INFO_ << "docs rows: " << docs->GetRows() << ", cols: " << docs->GetDim() << std::endl;
//     // LOG_KNOWHERE_INFO_ << "queries rows: " << queries->GetRows() << ", cols: " << queries->GetDim() << std::endl;

//     auto doc_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(docs->GetTensor());
//     auto doc_rows = docs->GetRows();
//     auto query_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(queries->GetTensor());
//     auto query_rows = queries->GetRows();

//     // key: dim, value: list of vec ids and values
//     std::unordered_map<knowhere::sparse::table_t, std::vector<knowhere::sparse::SparseIdVal<float>>> inverted_lut;
//     std::vector<float> row_sums(doc_rows, 0.0f);

//     for (int i = 0; i < doc_rows; i++) {
//         auto& doc = doc_tensor[i];
//         for (int j = 0; j < doc.size(); j++) {
//             auto [dim, value] = doc[j];
//             inverted_lut[dim].emplace_back(i, value);
//             row_sums[i] += value;
//         }
//     }
//     float avgdl = std::accumulate(row_sums.begin(), row_sums.end(), 0.0f) / doc_rows;
//     auto k1 = 1.2f;
//     auto b = 0.75f;

//     std::vector<float> ttl_scores(query_rows, 0.0f);

//     auto now = std::chrono::system_clock::now();
//     auto search_pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
//     std::vector<folly::Future<folly::Unit>> futs;
//     futs.reserve(query_rows);

//     for (int i = 0; i < query_rows; i++) {
//         futs.emplace_back(search_pool->push([&, i]() {
//             std::vector<float> scores(doc_rows, 0.0f);
//             auto& query = query_tensor[i];
//             for (int j = 0; j < query.size(); j++) {
//                 auto [dim, value] = query[j];
//                 auto& ids_and_values = inverted_lut[dim];
//                 for (auto& [id, val] : ids_and_values) {
//                     // value is idf, val is tf
//                     // compute tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)));
//                     float val_sum = row_sums[id];
//                     float score = val * value * (k1 + 1) / (val + k1 * (1 - b + b * (val_sum / avgdl)));
//                     scores[id] += score;
//                 }
//             }
//             float query_score = 0.0f;
//             for (int j = 0; j < doc_rows; j++) {
//                 query_score += scores[j];
//             }
//             ttl_scores[i] = query_score;
//         }));
//     }
//     knowhere::WaitAllSuccess(futs);
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
//     LOG_KNOWHERE_INFO_ << "Basic compute BM25 time: " << duration.count() / 1000 << "s" << std::endl;

//     auto sum_scores = std::accumulate(ttl_scores.begin(), ttl_scores.end(), 0.0f);
//     LOG_KNOWHERE_INFO_ << "sum scores: " << sum_scores << std::endl << std::endl << std::endl;
// }

// TEST_CASE("Test Mem Sparse Index MsMarco BM25 Compute BM25 basic slight optimization", "[float metrics]") {
//     LOG_KNOWHERE_INFO_ << "start slight optimization" << std::endl;
//     auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_document_bm25.csr");
//     auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_query_bm25.csr");

//     // auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_collection.csr");
//     // auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_query.csr");
//     // LOG_KNOWHERE_INFO_ << "docs rows: " << docs->GetRows() << ", cols: " << docs->GetDim() << std::endl;
//     // LOG_KNOWHERE_INFO_ << "queries rows: " << queries->GetRows() << ", cols: " << queries->GetDim() << std::endl;

//     auto doc_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(docs->GetTensor());
//     auto doc_rows = docs->GetRows();
//     auto query_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(queries->GetTensor());
//     auto query_rows = queries->GetRows();

//     // key: dim, value: list of vec ids and values
//     std::unordered_map<knowhere::sparse::table_t, std::vector<knowhere::sparse::SparseIdVal<float>>> inverted_lut;
//     std::vector<float> row_sums(doc_rows, 0.0f);

//     for (int i = 0; i < doc_rows; i++) {
//         auto& doc = doc_tensor[i];
//         for (int j = 0; j < doc.size(); j++) {
//             auto [dim, value] = doc[j];
//             inverted_lut[dim].emplace_back(i, value);
//             row_sums[i] += value;
//         }
//     }
//     float avgdl = std::accumulate(row_sums.begin(), row_sums.end(), 0.0f) / doc_rows;
//     auto k1 = 1.2f;
//     auto b = 0.75f;

//     auto one_minus_b = 1 - b;
//     auto k1_b = k1 * b;
//     auto k1_one_minus_b = k1 * one_minus_b;
//     auto k1_plus_one = k1 + 1;

//     std::vector<float> ttl_scores(query_rows, 0.0f);

//     auto now = std::chrono::system_clock::now();
//     auto search_pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
//     std::vector<folly::Future<folly::Unit>> futs;
//     futs.reserve(query_rows);

//     for (int i = 0; i < query_rows; i++) {
//         futs.emplace_back(search_pool->push([&, i]() {
//             std::vector<float> scores(doc_rows, 0.0f);
//             auto& query = query_tensor[i];
//             for (int j = 0; j < query.size(); j++) {
//                 auto [dim, value] = query[j];
//                 auto& ids_and_values = inverted_lut[dim];
//                 for (auto& [id, val] : ids_and_values) {
//                     // value is idf, val is tf
//                     // compute tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)));
//                     float val_sum = row_sums[id];
//                     float score = val * value * k1_plus_one / (val + k1_one_minus_b +  k1_b * (val_sum / avgdl));
//                     scores[id] += score;
//                 }
//             }
//             ttl_scores[i] = std::accumulate(scores.begin(), scores.end(), 0.0f);
//         }));
//     }
//     knowhere::WaitAllSuccess(futs);
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
//     LOG_KNOWHERE_INFO_ << "Slightly optimized compute BM25 time: " << duration.count() / 1000 << "s" << std::endl;

//     auto sum_scores = std::accumulate(ttl_scores.begin(), ttl_scores.end(), 0.0f);
//     LOG_KNOWHERE_INFO_ << "sum scores: " << sum_scores << std::endl << std::endl << std::endl;
// }

// TEST_CASE("Test Mem Sparse Index MsMarco BM25 Compute BM25 basic deep optimization", "[float metrics]") {
//     LOG_KNOWHERE_INFO_ << "start deep optimization" << std::endl;
//     auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_document_bm25.csr");
//     auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_passage_all_query_bm25.csr");

//     // auto docs = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_collection.csr");
//     // auto queries = LoadSparseEmbeddings("/home/zilliz/datasets/sparse-custom/ms_marco_tf_idf/milvus_query.csr");
//     // LOG_KNOWHERE_INFO_ << "docs rows: " << docs->GetRows() << ", cols: " << docs->GetDim() << std::endl;
//     // LOG_KNOWHERE_INFO_ << "queries rows: " << queries->GetRows() << ", cols: " << queries->GetDim() << std::endl;

//     auto doc_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(docs->GetTensor());
//     auto doc_rows = docs->GetRows();
//     auto query_tensor = static_cast<const knowhere::sparse::SparseRow<float>*>(queries->GetTensor());
//     auto query_rows = queries->GetRows();

//     std::unordered_map<knowhere::sparse::table_t, std::vector<knowhere::sparse::SparseIdVal<std::pair<float, float>>>> inverted_lut;
//     std::vector<float> row_sums(doc_rows, 0.0f);

//     for (int i = 0; i < doc_rows; i++) {
//         auto& doc = doc_tensor[i];
//         for (int j = 0; j < doc.size(); j++) {
//             auto [dim, value] = doc[j];
//             row_sums[i] += value;
//         }
//     }

//     auto k1 = 1.2f;
//     auto b = 0.75f;
//     for (int i = 0; i < doc_rows; i++) {
//         auto& doc = doc_tensor[i];
//         for (int j = 0; j < doc.size(); j++) {
//             auto [dim, value] = doc[j];
//             inverted_lut[dim].emplace_back(i, std::make_pair(1 + (k1 - k1 * b) / value, k1 * b * row_sums[i] / value));
//         }
//     }

//     float avgdl = std::accumulate(row_sums.begin(), row_sums.end(), 0.0f) / doc_rows;
//     float avgdl_inv = 1 / avgdl;

//     auto one_minus_b = 1 - b;
//     auto k1_b = k1 * b;
//     auto k1_one_minus_b = k1 * one_minus_b;
//     auto k1_plus_one = k1 + 1;

//     std::vector<float> ttl_scores(query_rows, 0.0f);

//     auto now = std::chrono::system_clock::now();

//     auto search_pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
//     std::vector<folly::Future<folly::Unit>> futures;
//     futures.reserve(query_rows);

//     for (int i = 0; i < query_rows; i++) {
//         futures.emplace_back(search_pool->push([&, i]() {
//             std::vector<float> scores(doc_rows, 0.0f);
//             auto& query = query_tensor[i];
//             for (int j = 0; j < query.size(); j++) {
//                 auto [dim, value] = query[j];
//                 auto& ids_and_values = inverted_lut[dim];
//                 for (auto& [id, val] : ids_and_values) {
//                     auto [x, y] = val;
//                     float score = value / (x + y * avgdl_inv);
//                     scores[id] += score;
//                 }
//             }
//             float query_score = 0.0f;
//             for (int j = 0; j < doc_rows; j++) {
//                 query_score += scores[j];
//             }
//             ttl_scores[i] = query_score;
//         }));
//     }

//     knowhere::WaitAllSuccess(futures);
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now);
//     LOG_KNOWHERE_INFO_ << "Deep optimized compute BM25 time: " << duration.count() / 1000 << "s" << std::endl;

//     auto sum_scores = std::accumulate(ttl_scores.begin(), ttl_scores.end(), 0.0f) * k1_plus_one;
//     LOG_KNOWHERE_INFO_ << "sum scores: " << sum_scores << std::endl << std::endl << std::endl;
// }

/**
Single thread:

I0914 12:19:42.408084 2871986 test_sparse_bench.cc:171] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] start basic IP
I0914 12:19:53.249378 2871986 test_sparse_bench.cc:177] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:19:53.249405 2871986 test_sparse_bench.cc:178] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:23:25.774439 2871986 test_sparse_bench.cc:214] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] compute IP time: 201s
I0914 12:23:25.774497 2871986 test_sparse_bench.cc:216] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] sum scores: 2.64906e+08

I0914 12:23:27.340260 2871986 test_sparse_bench.cc:221] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] start basic
I0914 12:23:37.291755 2871986 test_sparse_bench.cc:227] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:23:37.291781 2871986 test_sparse_bench.cc:228] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:27:23.981349 2871986 test_sparse_bench.cc:274] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] Basic compute BM25 time: 215s
I0914 12:27:23.981388 2871986 test_sparse_bench.cc:277] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] sum scores: 3.37565e+08


I0914 12:27:25.375751 2871986 test_sparse_bench.cc:281] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] start slight optimization
I0914 12:27:35.169090 2871986 test_sparse_bench.cc:287] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:27:35.169116 2871986 test_sparse_bench.cc:288] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:31:17.305440 2871986 test_sparse_bench.cc:339] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] Slightly optimized compute BM25 time: 211s
I0914 12:31:17.305480 2871986 test_sparse_bench.cc:342] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] sum scores: 3.37565e+08


I0914 12:31:18.698810 2871986 test_sparse_bench.cc:346] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] start deep optimization
I0914 12:31:28.494100 2871986 test_sparse_bench.cc:352] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:31:28.494127 2871986 test_sparse_bench.cc:353] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:34:53.200837 2871986 test_sparse_bench.cc:412] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] Slightly optimized compute BM25 time: 193s
I0914 12:34:53.200874 2871986 test_sparse_bench.cc:415] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] sum scores: 1.53439e+08

Thread pool:

I0914 12:35:26.264236 2874051 test_sparse_bench.cc:181] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] start basic IP
I0914 12:35:37.671638 2874051 test_sparse_bench.cc:187] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:35:37.671666 2874051 test_sparse_bench.cc:188] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:35:48.168426 2874051 thread_pool.h:179] [KNOWHERE][InitGlobalSearchThreadPool][knowhere_tests] Init global search thread pool with size 12
I0914 12:36:33.668438 2874051 test_sparse_bench.cc:231] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] compute IP time: 45s
I0914 12:36:33.668479 2874051 test_sparse_bench.cc:233] [KNOWHERE][CATCH2_INTERNAL_TEST_12][knowhere_tests] sum scores: 2.64906e+08


I0914 12:36:35.114395 2874051 test_sparse_bench.cc:238] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] start basic
I0914 12:36:45.543901 2874051 test_sparse_bench.cc:244] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:36:45.543928 2874051 test_sparse_bench.cc:245] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:37:40.944187 2874051 test_sparse_bench.cc:299] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] Basic compute BM25 time: 45s
I0914 12:37:40.944226 2874051 test_sparse_bench.cc:302] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] sum scores: 3.37565e+08


I0914 12:37:42.386799 2874051 test_sparse_bench.cc:306] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] start slight optimization
I0914 12:37:52.773509 2874051 test_sparse_bench.cc:312] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:37:52.773536 2874051 test_sparse_bench.cc:313] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:38:47.887728 2874051 test_sparse_bench.cc:368] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] Slightly optimized compute BM25 time: 45s
I0914 12:38:47.887769 2874051 test_sparse_bench.cc:371] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] sum scores: 3.37565e+08


I0914 12:38:49.350993 2874051 test_sparse_bench.cc:375] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] start deep optimization
I0914 12:38:59.475908 2874051 test_sparse_bench.cc:381] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] docs rows: 8841823, cols: 4069890
I0914 12:38:59.475934 2874051 test_sparse_bench.cc:382] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] queries rows: 6980, cols: 3791524
I0914 12:39:54.109275 2874051 test_sparse_bench.cc:449] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] Slightly optimized compute BM25 time: 43s
I0914 12:39:54.109313 2874051 test_sparse_bench.cc:452] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] sum scores: 1.53439e+08

Multi thread:

I0914 15:21:50.690923 2907373 test_sparse_bench.cc:253] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] compute IP time: 43s
I0914 15:21:50.690965 2907373 test_sparse_bench.cc:255] [KNOWHERE][CATCH2_INTERNAL_TEST_14][knowhere_tests] sum scores: 2.64906e+08

I0914 15:23:01.202868 2907373 test_sparse_bench.cc:321] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] Basic compute BM25 time: 48s
I0914 15:23:01.202911 2907373 test_sparse_bench.cc:324] [KNOWHERE][CATCH2_INTERNAL_TEST_16][knowhere_tests] sum scores: 3.37565e+08

I0914 15:25:14.194442 2907373 test_sparse_bench.cc:471] [KNOWHERE][CATCH2_INTERNAL_TEST_20][knowhere_tests] Deep optimized compute BM25 time: 42s
I0914 15:25:14.194487 2907373 test_sparse_bench.cc:474] [KNOWHERE][CATCH2_INTERNAL_TEST_20][knowhere_tests] sum scores: 3.37565e+08

I0914 15:24:09.288750 2907373 test_sparse_bench.cc:390] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] Slightly optimized compute BM25 time: 46s
I0914 15:24:09.288818 2907373 test_sparse_bench.cc:393] [KNOWHERE][CATCH2_INTERNAL_TEST_18][knowhere_tests] sum scores: 3.37565e+08



Search with index:

使用的 avgdl 越大，搜索越快。因为更大的 avgdl 会导致 WAND 认为的 max score 越大，从而跳过更多的文档

average max score used in WAND:

* avgdl = 100: 1.31
* avgdl = 12.7782: 0.7717

I0918 19:05:40.903488 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric IP topk 10 avgdl 100 refine_factor 1]
I0918 19:05:40.903618 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:05:56.659610 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:06:39.300097 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:06:39.300134 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:06:39.300474 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 42640ms
I0918 19:06:40.223002 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:07:02.251252 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.540422
I0918 19:07:02.384202 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:07:02.646298 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 4519.13
I0918 19:07:02.646333 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 4902.27
I0918 19:07:02.646811 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 262ms
I0918 19:07:03.867897 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:07:03.868005 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric BM25 topk 10 avgdl 100 refine_factor 1]
I0918 19:07:03.868016 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:07:18.436662 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:07:56.775480 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:07:56.775517 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:07:56.775844 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 38339ms
I0918 19:07:57.664047 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:08:20.461113 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 1.31434
I0918 19:08:20.592243 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:08:20.933734 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 4336.58
I0918 19:08:20.933774 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 4696.94
I0918 19:08:20.934454 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 342ms
I0918 19:08:22.139057 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:08:32.811997 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric IP topk 10 avgdl 100 refine_factor 10]
I0918 19:08:32.812031 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:08:47.071655 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:09:30.909574 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:09:30.909611 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:09:30.909945 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 43838ms
I0918 19:09:31.845590 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:09:54.257788 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.540422
I0918 19:09:54.389516 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:09:54.853351 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 11239.6
I0918 19:09:54.853386 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 11665
I0918 19:09:54.854113 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 464ms
I0918 19:09:56.092664 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:09:56.092759 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric BM25 topk 10 avgdl 100 refine_factor 10]
I0918 19:09:56.092770 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:10:10.539947 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:10:49.353758 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:10:49.353796 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:10:49.354110 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 38814ms
I0918 19:10:50.274036 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:11:13.677614 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 1.31434
I0918 19:11:13.816243 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:11:14.515808 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 10122.7
I0918 19:11:14.515842 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 10531.7
I0918 19:11:14.516534 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 100] Index search time: 700ms
I0918 19:11:15.776207 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:11:27.006958 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric IP topk 10 avgdl 12.7782 refine_factor 1]
I0918 19:11:27.006991 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:11:42.617060 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:12:27.342286 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:12:27.342325 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:12:27.342662 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 44725ms
I0918 19:12:28.273334 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:12:50.316398 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.540422
I0918 19:12:50.447885 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:12:50.706691 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 4519.13
I0918 19:12:50.706728 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 4902.27
I0918 19:12:50.707455 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 259ms
I0918 19:12:51.912231 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:12:51.912329 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric BM25 topk 10 avgdl 12.7782 refine_factor 1]
I0918 19:12:51.912341 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:13:06.553443 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:13:45.393476 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:13:45.393514 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:13:45.393841 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 38840ms
I0918 19:13:46.302263 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:14:09.012310 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.771745
I0918 19:14:09.143406 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:14:09.699379 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 9398.09
I0918 19:14:09.699416 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 9832.53
I0918 19:14:09.700085 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 556ms
I0918 19:14:10.913389 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:14:21.531868 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric IP topk 10 avgdl 12.7782 refine_factor 10]
I0918 19:14:21.531899 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:14:35.812780 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:15:18.703886 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:15:18.703924 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:15:18.704243 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 42891ms
I0918 19:15:19.635396 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:15:41.557834 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.540422
I0918 19:15:41.690737 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:15:42.148675 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 11239.6
I0918 19:15:42.148715 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 11665
I0918 19:15:42.149026 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 458ms
I0918 19:15:43.376839 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]

I0918 19:15:43.376945 3221957 test_sparse_bench.cc:161] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] [metric BM25 topk 10 avgdl 12.7782 refine_factor 10]
I0918 19:15:43.376956 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_INVERTED_INDEX
I0918 19:15:57.887537 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:16:40.696674 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 0
I0918 19:16:40.696714 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 0
I0918 19:16:40.697052 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 42809ms
I0918 19:16:41.728327 3221957 test_sparse_bench.cc:164] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]      Index: SPARSE_WAND
I0918 19:17:05.261098 3221957 sparse_inverted_index.h:262] [KNOWHERE][Add][knowhere_tests] Average of max scores in dimensions: 0.771745
I0918 19:17:05.393265 3221957 test_sparse_bench.cc:186] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests] start search
I0918 19:17:06.564471 3221957 sparse_index_node.cc:123] [KNOWHERE][Search][knowhere_tests]              Average push count: 23211.8
I0918 19:17:06.564509 3221957 sparse_index_node.cc:124] [KNOWHERE][Search][knowhere_tests]              Average score count: 23653.4
I0918 19:17:06.565140 3221957 test_sparse_bench.cc:192] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]             [ 0.32, 0.6, 12.7782] Index search time: 1171ms
I0918 19:17:07.845490 3221957 test_sparse_bench.cc:196] [KNOWHERE][CATCH2_INTERNAL_TEST_10][knowhere_tests]
 */