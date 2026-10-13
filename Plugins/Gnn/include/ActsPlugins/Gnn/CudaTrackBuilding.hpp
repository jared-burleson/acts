// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <memory>

namespace ActsPlugins {
/// @addtogroup gnn_plugin
/// @{

/// Track building implementation using CUDA
class CudaTrackBuilding final : public TrackBuildingBase {
 public:
  /// Configuration for CUDA track building
  struct Config {
    /// Use one block implementation
    bool useOneBlockImplementation = true;
    /// Do junction removal
    bool doJunctionRemoval = false;
    /// Minimum candidate size
    std::size_t minCandidateSize = 3;
  };

  /// Constructor
  /// @param cfg Configuration object
  /// @param logger Logger instance
  CudaTrackBuilding(const Config &cfg,
                    std::unique_ptr<const Acts::Logger> logger)
      : m_cfg(cfg), m_logger(std::move(logger)) {}

  std::vector<std::vector<int>> operator()(
      PipelineTensors tensors, std::vector<int> &spacePointIDs,
      const ExecutionContext &execContext = {}) override;

  /// Track candidates in device memory
  struct DeviceCandidates {
    /// Space point ids grouped by candidate (device, numSpacePoints entries)
    Tensor<int> spacePointIds;
    /// Candidate i is spacePointIds[bounds[i], bounds[i + 1]) (device,
    /// numCandidates + 1 valid entries)
    Tensor<int> bounds;
    /// Number of candidates (connected components), without the
    /// minCandidateSize selection
    std::size_t numCandidates = 0;
  };

  /// Track building with the space point ids and the result on the device
  ///
  /// @param tensors Input pipeline tensors
  /// @param cudaSpacePointIds IDs of the nodes (device pointer with n_nodes
  ///        entries, ready on the stream of @p execContext)
  /// @param numSpacePoints Number of nodes
  /// @param execContext Device & stream information
  /// @return The candidates, ready on the stream of @p execContext
  DeviceCandidates buildCandidates(PipelineTensors tensors,
                                   const int *cudaSpacePointIds,
                                   std::size_t numSpacePoints,
                                   const ExecutionContext &execContext);
  /// Get configuration
  /// @return Configuration object
  const Config &config() const { return m_cfg; }

 private:
  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
  const auto &logger() const { return *m_logger; }
};

/// @}
}  // namespace ActsPlugins
