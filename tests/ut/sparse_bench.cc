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

#include <iostream>
#include <future>
#include <chrono>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"
#include "wand.h"


TEST_CASE("SPARSE knowheresparse") {
    using std::make_tuple;

    const auto train_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/base_1M.csr");
    auto nb = train_ds->GetRows();
    const auto query_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/queries.dev.csr");

    auto [drop_ratio_build, drop_ratio_search] = GENERATE(table<float, float>({
        // {0.0, 0.0},
        {0.32, 0.0},
        {0.32, 0.6},
    }));

    auto max_q_dim = GENERATE(size_t(5), size_t(10), size_t(20));

    auto topk = 10;
    auto dim = train_ds->GetDim();

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_build = drop_ratio_build,
                                      drop_ratio_search = drop_ratio_search,
                                      max_q_dim = max_q_dim]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::MAX_Q_DIM] = max_q_dim;
        return json;
    };

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };

    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        // make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
    }));

    std::cout << "======== " << name << " ========\n";
    std::cout << "drop_ratio_build: " << drop_ratio_build << ", drop_ratio_search: " << drop_ratio_search << ", max_q_dim: " << max_q_dim << "\n";

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();

    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    auto now = std::chrono::high_resolution_clock::now();
    REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now);
    std::cout << "Build time: " << duration.count() << "ms\n";
    REQUIRE(idx.Size() > 0);
    REQUIRE(idx.Count() == nb);

    now = std::chrono::high_resolution_clock::now();
    auto res = idx.Search(*query_ds, json, nullptr);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now);
    auto qps = query_ds->GetRows() * 1000.0 / duration.count();
    std::cout << "QPS: " << qps << "\n";

    auto gt = ReadSparseGT("/home/zilliz/datasets/sparse-bigann-23/base_1M.dev.gt");
    float recall = GetKNNRecall(*gt, *res.value());
    std::cout << "Recall: " << recall << "\n";
    std::cout << "======== " << name << " END ========\n";
}

TEST_CASE("SPARSE pyanns") {
    using std::make_tuple;

    auto [drop_ratio_build, drop_ratio_search] = GENERATE(table<float, float>({
        // {0.0, 0.0},
        {0.32, 0.0},
        {0.32, 0.6},
    }));
    auto max_q_dim = GENERATE(size_t(5), size_t(10), size_t(20));

    auto topk = 10;

    pyanns::IndexSparse index;
    std::cout << "======== " << "pyanns" << " ========\n";
    std::cout << "drop_ratio_build: " << drop_ratio_build << ", drop_ratio_search: " << drop_ratio_search << ", max_q_dim: " << max_q_dim << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    index.add("/home/zilliz/datasets/sparse-bigann-23/base_1M.csr", drop_ratio_build);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "Build time: " << duration.count() << "ms\n";
    start = std::chrono::high_resolution_clock::now();
    auto res = index.search_batch_file("/home/zilliz/datasets/sparse-bigann-23/queries.dev.csr", topk, drop_ratio_search, 10, max_q_dim);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    auto qps = 6980 * 1000.0 / duration.count();
    std::cout << "QPS: " << qps << "\n";

    auto gt = ReadSparseGT("/home/zilliz/datasets/sparse-bigann-23/base_1M.dev.gt");
    float recall = GetKNNRecall(*gt, res);
    free(res);
    std::cout << "Recall: " << recall << "\n";
    std::cout << "======== " << "pyanns" << " END ========\n";

}

TEST_CASE("SPARSE sparse_bench") {
    auto [drop_ratio_build, drop_ratio_search] = GENERATE(table<float, float>({
        // {0.0, 0.0},
        {0.32, 0.0},
        {0.32, 0.6},
    }));
    pyanns::IndexSparse index;
    index.add("/home/zilliz/datasets/sparse-bigann-23/base_1M.csr", drop_ratio_build);


    using std::make_tuple;

    const auto train_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/base_1M.csr");
    auto nb = train_ds->GetRows();
    const auto query_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/queries.dev.csr");



    auto max_q_dim = GENERATE(size_t(5), size_t(10), size_t(20));

    auto topk = 10;
    auto dim = train_ds->GetDim();

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_build = drop_ratio_build,
                                      drop_ratio_search = drop_ratio_search,
                                      max_q_dim = max_q_dim]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::MAX_Q_DIM] = max_q_dim;
        return json;
    };

    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        // make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
    }));

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();

    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    auto now = std::chrono::high_resolution_clock::now();
    REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

    auto res = idx.Search(*query_ds, json, nullptr);
}
