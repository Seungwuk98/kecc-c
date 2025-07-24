#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

namespace kecc {

using TypeID = mlir::TypeID;

} // namespace kecc

#define DECLARE_KECC_TYPE_ID(T) MLIR_DECLARE_EXPLICIT_TYPE_ID(T)
#define DEFINE_KECC_TYPE_ID(T) MLIR_DEFINE_EXPLICIT_TYPE_ID(T)
