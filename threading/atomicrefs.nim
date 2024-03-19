
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
  var timpl, tname: NimNode
  if tp.kind == nnkSym:
    timpl = tp.getImpl()
    timpl.expectKind(nnkTypeDef)
    tname = tp
  elif tp.kind == nnkRefTy:
    timpl = tp[^1].getImpl()
    tname = tp

  result = newStmtList()

  var tobj = timpl[^1]
  if tobj.kind == nnkRefTy:
    tobj = tobj[0]

  var tbody = tobj
  if tbody.kind == nnkSym:
    let ity = tbody.getImpl()
    ity.expectKind(nnkTypeDef)
    tbody = ity[^1]

  tbody.expectKind(nnkObjectTy)

  let idents = tbody[^1]
  idents.expectKind(nnkRecList)
  for ident in idents:
    if ident[0].kind != nnkPostFix:
      continue
    let name = ident(ident[0][1].strVal)
    let fieldName = ident ident[0][1].repr
    let fieldTp = ident[1]
    let obj = ident "obj"
    let acc = quote do:
      when `fieldTp` is ref:
        proc `name`*(`obj`: Atomic[`tname`]): Atomic[`fieldTp`] =
          newAtomicRef(`obj`.unsafeGet().`fieldName`)
        atomicAccessors(`fieldTp`)
      else:
        proc `name`*(`obj`: Atomic[`tname`]): `fieldTp` =
          `obj`.unsafeGet().`fieldName`
    result.add acc
  echo "RES:\n", result.repr

when isMainModule:
  type
    Test* = ref object
      msg*: string

    Foo* = ref object
      inner*: Test

  # expandMacros:
  atomicAccessors(Foo)

  type
    Test2* = object
      msg2*: string
    TestRef* = ref Test2

    Foo2* = object
      inner2*: ref Test2
    FooRef* = ref Foo2

  # expandMacros:
  # atomicAccessors(FooRef)

  atomicAccessors(ref Foo2)
