/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hip;
using namespace mlir::bufferization;

#include "HipDialect.cpp.inc"

void HipDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HipTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "HipOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AllocationOpInterface for Hip_AllocOp
//===----------------------------------------------------------------------===//

std::optional<Operation *> mlir::hip::AllocOp::buildDealloc(
    OpBuilder &builder, Value alloc) {
  // Extract handle from the alloc operation
  auto allocOp = alloc.getDefiningOp<hip::AllocOp>();
  if (!allocOp)
    return std::nullopt;

  return builder.create<hip::FreeOp>(
      alloc.getLoc(),
      allocOp.getHandle(),  // Same context from the alloc operation
      alloc                 // Buffer to free
  ).getOperation();
}

std::optional<Value> mlir::hip::AllocOp::buildClone(
    OpBuilder &builder, Value alloc) {
  // GPU buffer cloning is complex, let MLIR handle via explicit copies
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface Implementations
//===----------------------------------------------------------------------===//

void mlir::hip::AllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // hip.alloc allocates GPU memory
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::FreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // hip.free deallocates GPU memory
  effects.emplace_back(MemoryEffects::Free::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ConvOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Read inputs (conservative: assumes reads from memory)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  // Write output
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Read inputs and write output (result is read-write due to beta != 0)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::MaxPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::AvgPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GetConstantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Allocates view to constant memory (read-only effect)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// Type and op class implementations (parse/print/verify, TypeIDs)
#define GET_TYPEDEF_CLASSES
#include "HipTypes.cpp.inc"

#define GET_OP_CLASSES
#include "HipOps.cpp.inc"
