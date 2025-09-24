#ifndef KECC_C_TYPE_CONVERTER_H
#define KECC_C_TYPE_CONVERTER_H

#include "kecc/c/Clang.h"
#include "kecc/c/Diag.h"
#include "kecc/c/Visitor.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/Type.h"

namespace kecc::c {

class RecordDeclManager;

class TypeConverter : public TypeVisitor<TypeConverter, ir::Type> {
public:
  TypeConverter(ClangDiagManager *diag, ir::IRContext *ctx,
                RecordDeclManager *recordDeclMgr)
      : TypeVisitor(diag), ctx(ctx), recordDeclMgr(recordDeclMgr) {}

  ir::Type VisitQualType(QualType qt, const SourceLocation &loc);
  ir::Type VisitArrayType(const ArrayType *T, const SourceLocation &loc);
  ir::Type VisitRecordType(const RecordType *T, const SourceLocation &loc);
  ir::Type VisitTypedefType(const TypedefType *T, const SourceLocation &loc);
  ir::Type VisitPointerType(const PointerType *T, const SourceLocation &loc);
  ir::Type VisitFunctionType(const FunctionType *T, const SourceLocation &loc);
  ir::Type VisitDecayedType(const DecayedType *T, const SourceLocation &loc);
  ir::Type VisitBuiltinType(const BuiltinType *T, const SourceLocation &loc);
  ir::Type VisitParenType(const ParenType *T, const SourceLocation &loc);
  ir::Type VisitElaboratedType(const ElaboratedType *T,
                               const SourceLocation &loc);

private:
  RecordDeclManager *recordDeclMgr;
  ir::IRContext *ctx;
};

} // namespace kecc::c

#endif // KECC_C_TYPE_CONVERTER_H
