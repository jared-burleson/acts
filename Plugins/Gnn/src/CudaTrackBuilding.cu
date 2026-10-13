// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/Zip.hpp"
#include "ActsPlugins/Gnn/CudaTrackBuilding.hpp"
#include "ActsPlugins/Gnn/detail/ConnectedComponents.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/JunctionRemoval.hpp"

using namespace Acts;

namespace ActsPlugins {

CudaTrackBuilding::DeviceCandidates CudaTrackBuilding::buildCandidates(
    PipelineTensors tensors, const int* cudaSpacePointIds,
    std::size_t numSpacePoints, const ExecutionContext& execContext) {
  ACTS_VERBOSE("Start CUDA track building");
  if (!(tensors.edgeIndex.device().isCuda() &&
        tensors.edgeScores.value().device().isCuda())) {
    throw std::runtime_error(
        "CudaTrackBuilding expects tensors to be on CUDA!");
  }

  auto numEdges = static_cast<std::size_t>(tensors.edgeIndex.shape().at(1));

  // At least one entry, so that the tensors are valid also without candidates
  DeviceCandidates result{
      Tensor<int>::Create({std::max<std::size_t>(numSpacePoints, 1), 1},
                          execContext),
      Tensor<int>::Create({numSpacePoints + 1, 1}, execContext), 0};

  if (numEdges == 0) {
    ACTS_DEBUG("No edges remained after edge classification");
    return result;
  }

  auto stream = execContext.stream.value();

  auto cudaSrcPtr = tensors.edgeIndex.data();
  auto cudaTgtPtr = tensors.edgeIndex.data() + numEdges;

  // Everything below is enqueued without host synchronization; the edge and
  // label counts stay in device memory until the results are copied back.
  // counters[0]: number of edges after junction removal, [1]: number of labels
  int* cudaCounters{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&cudaCounters, 2 * sizeof(int), stream));
  const int* cudaNumEdges = nullptr;

  std::int64_t* cudaJrEdges{};
  if (m_cfg.doJunctionRemoval) {
    assert(tensors.edgeScores->shape().at(0) ==
           tensors.edgeIndex.shape().at(1));
    ACTS_DEBUG("Do junction removal...");
    ACTS_CUDA_CHECK(cudaMallocAsync(
        &cudaJrEdges, 2 * numEdges * sizeof(std::int64_t), stream));
    detail::junctionRemovalCudaAsync(
        numEdges, numSpacePoints, tensors.edgeScores->data(), cudaSrcPtr,
        cudaTgtPtr, cudaJrEdges, cudaJrEdges + numEdges, cudaCounters,
        stream);
    cudaSrcPtr = cudaJrEdges;
    cudaTgtPtr = cudaJrEdges + numEdges;
    cudaNumEdges = cudaCounters;
  }

  int* cudaLabels{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&cudaLabels, numSpacePoints * sizeof(int), stream));

  detail::connectedComponentsCudaAsync(numEdges, cudaNumEdges, cudaSrcPtr,
                                       cudaTgtPtr, numSpacePoints, cudaLabels,
                                       cudaCounters + 1, stream);

  // Sort the space point IDs by label and compute the bounds of each label
  detail::findTrackCandidateBoundsAsync(
      cudaLabels, cudaSpacePointIds, result.spacePointIds.data(),
      result.bounds.data(), numSpacePoints, cudaCounters + 1, stream);

  int counters[2]{};
  ACTS_CUDA_CHECK(cudaMemcpyAsync(counters, cudaCounters, sizeof(counters),
                                  cudaMemcpyDeviceToHost, stream));

  // Free Memory
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaLabels, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaCounters, stream));
  if (m_cfg.doJunctionRemoval) {
    ACTS_CUDA_CHECK(cudaFreeAsync(cudaJrEdges, stream));
  }

  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  ACTS_CUDA_CHECK(cudaGetLastError());

  if (m_cfg.doJunctionRemoval) {
    ACTS_DEBUG("Removed " << numEdges - counters[0]
                          << " edges in junction removal");
    if (counters[0] == 0) {
      ACTS_WARNING(
          "No edges remained after junction removal, this should not happen!");
      return result;
    }
  }
  result.numCandidates = static_cast<std::size_t>(counters[1]);
  ACTS_VERBOSE("Found " << result.numCandidates << " track candidates");
  return result;
}

std::vector<std::vector<int>> CudaTrackBuilding::operator()(
    PipelineTensors tensors, std::vector<int>& spacePointIds,
    const ExecutionContext& execContext) {
  const auto numSpacePoints = spacePointIds.size();
  auto stream = execContext.stream.value();

  int* cudaSpacePointIds{};
  ACTS_CUDA_CHECK(cudaMallocAsync(
      &cudaSpacePointIds, std::max<std::size_t>(numSpacePoints, 1) * sizeof(int),
      stream));
  ACTS_CUDA_CHECK(cudaMemcpyAsync(cudaSpacePointIds, spacePointIds.data(),
                                  numSpacePoints * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
  auto candidates = buildCandidates(std::move(tensors), cudaSpacePointIds,
                                    numSpacePoints, execContext);
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaSpacePointIds, stream));

  const auto numberLabels = candidates.numCandidates;
  if (numberLabels == 0) {
    return {};
  }

  std::vector<int> bounds(numberLabels + 1);
  ACTS_CUDA_CHECK(cudaMemcpyAsync(bounds.data(), candidates.bounds.data(),
                                  (numberLabels + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaMemcpyAsync(
      spacePointIds.data(), candidates.spacePointIds.data(),
      numSpacePoints * sizeof(int), cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  ACTS_DEBUG("Bounds size: " << bounds.size());
  if (numberLabels >= 6) {
    ACTS_DEBUG("Bounds: " << bounds.at(0) << ", " << bounds.at(1) << ", "
                          << bounds.at(2) << ", ..., "
                          << bounds.at(numberLabels - 2) << ", "
                          << bounds.at(numberLabels - 1) << ", "
                          << bounds.at(numberLabels));
  } else {
    ACTS_DEBUG("Bounds: " << [&]() {
      std::stringstream ss;
      ss << bounds.at(0);
      for (std::size_t i = 1; i <= numberLabels; ++i) {
        ss << ", " << bounds.at(i);
      }
      return ss.str();
    }());
  }

  std::vector<std::vector<int>> trackCandidates;
  trackCandidates.reserve(numberLabels);
  for (std::size_t label = 0ul; label < numberLabels; ++label) {
    int start = bounds.at(label);
    int end = bounds.at(label + 1);

    assert(start >= 0);
    assert(end <= static_cast<int>(numSpacePoints));
    assert(start <= end);

    if (end - start < m_cfg.minCandidateSize) {
      continue;
    }

    trackCandidates.emplace_back(spacePointIds.begin() + start,
                                 spacePointIds.begin() + end);
  }

  return trackCandidates;
}

}  // namespace ActsPlugins
