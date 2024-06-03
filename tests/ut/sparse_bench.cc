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

#include <chrono>
#include <future>
#include <iostream>

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

    const auto train_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/base_small.csr");
    auto nb = train_ds->GetRows();
    const auto query_ds = ReadSparseMatrix("/home/zilliz/datasets/sparse-bigann-23/queries.dev.csr");
    printf("FUCK! nb = %d\n", (int)nb);

    auto topk = 10;
    auto dim = train_ds->GetDim();

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;

        json[knowhere::indexparam::HNSW_M] = 32;
        json[knowhere::indexparam::EFCONSTRUCTION] = 120;
        json[knowhere::indexparam::EF] = 120;
        json[knowhere::indexparam::IS_SPARSE] = true;
        return json;
    };

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };

    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        // make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
        make_tuple(knowhere::IndexEnum::INDEX_HNSW, base_gen),
    }));

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();

    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    auto now = std::chrono::high_resolution_clock::now();
    REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now);
    std::cout << "Build time: " << duration.count() << "ms\n";
    REQUIRE(idx.Size() > 0);
    REQUIRE(idx.Count() == nb);

    now = std::chrono::high_resolution_clock::now();
    auto res = idx.Search(*query_ds, json, nullptr);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now);
    auto qps = query_ds->GetRows() * 1000.0 / duration.count();
    std::cout << "QPS: " << qps << "\n";

    auto gt = ReadSparseGT("/home/zilliz/datasets/sparse-bigann-23/base_small.dev.gt");
    float recall = GetKNNRecall(*gt, *res.value());
    std::cout << "Recall: " << recall << "\n";
    std::cout << "======== " << name << " END ========\n";
}