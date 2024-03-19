
import std/atomics

when defined(gcOrc):
  const
    rcIncrement = 0b10000 # so that lowest 4 bits are not touched
    rcMask = 0b1111
    rcShift = 4      # shift by rcShift to get the reference counter
else:
  const
    rcIncrement = 0b1000 # so that lowest 3 bits are not touched
    rcMask = 0b111
    rcShift = 3      # shift by rcShift to get the reference counter

type
  RefHeader = object
    rc: int # the object header is now a single RC field.
            # we could remove it in non-debug builds for the 'owned ref'
            # design but this seems unwise.
    when defined(gcOrc):
      rootIdx: int # thanks to this we can delete potential cycle roots
                   # in O(1) without doubly linked lists

  Cell = ptr RefHeader

template head(p: pointer): Cell =
  cast[Cell](cast[int](p) -% sizeof(RefHeader))
template count(x: Cell): untyped =
  x.rc shr rcShift

type
  Atomic*[T: ref] {.requiresInit.} = object
    rp {.cursor.}: T

proc `=destroy`*[T](aref: Atomic[T]) =
  if aref.rp != nil:
    var cell = head(cast[pointer](aref.rp))
    echo "destroy aref: ", cast[pointer](aref.rp).repr, " rc: ", cell.rc, " cnt: ", cell.count()
    # `atomicDec` returns the new value
    if atomicDec(cell.rc, rcIncrement) == -rcIncrement:
      echo "\nlast <<<"
      inc cell.rc, rcIncrement # hack to re-use normal destroy
      `=destroy`(aref.rp)
      echo ">> done"

proc `=copy`*[T](dest: var Atomic[T]; source: Atomic[T]) =
  echo "copy"
  # protect against self-assignments:
  if dest.rp != source.rp:
    `=destroy`(dest)
    wasMoved(dest)
    dest.rp = source.rp
    var cell = head(cast[pointer](dest.rp))
    discard atomicInc(cell.rc, rcIncrement)
    echo "copy cnt: ", cell.count

proc newAtomic*[T: ref](obj: sink T): Atomic[T] =
  result = Atomic[T](rp: move obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc newAtomicRef*[T: ref](obj: T): Atomic[T] =
  result = Atomic[T](rp: obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc unsafeGet*[T](aref: Atomic[T]): lent T =
  aref.rp

proc unsafeCount*[T](aref: ref T): int =
  var cell = head(cast[pointer](aref))
  cell.count()



import macros
import typetraits

macro atomicAccessors*(tp: typed) =

  tp.expectKind(nnkSym)
  let tname = ident tp.strVal
  let timpl = tp.getImpl()
  timpl.expectKind(nnkTypeDef)
  timpl[^1].expectKind(nnkRefTy)

  result = newStmtList()

  echo "TP: ", tname
  echo "TP:\n", timpl.treeRepr
  let tbody = timpl[^1][0]
  if tbody.kind == nnkObjectTy:
    let idents = tbody[^1]
    idents.expectKind(nnkRecList)
    echo "obj:\n", idents.treeRepr
    for ident in idents:
      if ident[0].kind != nnkPostFix:
        continue
      let name = ident(ident[0][1].repr & "Access")
      let fieldName = ident ident[0][1].repr
      let fieldTp = ident[1]
      let obj = ident "obj"
      echo "NAME: ", name
      let acc = quote do:
        when `fieldTp` is ref:
          proc `name`*(`obj`: Atomic[`tname`]): Atomic[`fieldTp`] =
            newAtomicRef(`obj`.unsafeGet().`fieldName`)
          atomicAccessors(`fieldTp`)
      result.add acc
  echo "RES:\n", result.treeRepr

# macro mkAccessor(name, tp, parentTp: untyped): untyped =
#   let n = ident name.strVal
#   let obj = ident "obj"

#   result = quote do:
#     proc `n`(`obj`: Atomic[`parentTp`]): Atomic[`tp`] =
#       newAtomicRef(`obj`.unsafeGet().`n`)
#     atomicAccessors(`tp`)

# template atomicAccessors*(tp: typed) =
#   for name, field in fieldPairs(tp()[]):
#     when typeof(field) is ref:
#       mkAccessor(name, typeof(field), tp)

when isMainModule:
  type
    Test* = ref object
      msg*: string

    Foo* = ref object
      inner*: Test

  atomicAccessors(Foo)
