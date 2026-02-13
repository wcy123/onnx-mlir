<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX HIP/DNN EP Documentation

Documentation for the ONNX Runtime Execution Provider with MLIR-based compilation for AMD GPUs.

## Getting Started

**New to the project?** Start here:
1. [Architecture Overview](design/ARCHITECTURE.md) - System design and major decisions
2. [MLIR Compilation Pipeline](design/MLIR-COMPILATION-OVERVIEW.md) - How compilation works
3. [Demo Walkthrough](guides/DEMO.md) - End-to-end example

## Documentation Sections

### [Design Documents](design/)
Architecture and design decisions for the entire system:
- System-wide architecture and compilation pipeline
- MLIR compilation pipeline details (module structure, passes, dialects)
- Feature-specific designs (constants, memory, dynamic shapes)
- Design alternatives and comparisons

### [Integration](integration/)
External dependency integration:
- ONNX-MLIR integration approach
- Integration status and build configuration

### [Guides](guides/)
Practical how-to documentation:
- Demo walkthroughs and presentations

### [Meta](meta/)
Documentation standards and processes:
- How to write design documents
- Documentation templates and guidelines

## Quick Links

- **For New Contributors:** [ARCHITECTURE.md](design/ARCHITECTURE.md)
- **For MLIR Developers:** [mlir/README.md](design/mlir/README.md)
- **For Demo Walkthrough:** [DEMO.md](guides/DEMO.md)
- **For Writing Docs:** [HOW-TO-WRITE-DESIGN-DOCS.md](meta/HOW-TO-WRITE-DESIGN-DOCS.md)
