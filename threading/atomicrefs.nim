
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
  Atomic*[T] {.requiresInit.} = object
    when T is ref:
      rp {.cursor.}: T
    elif T is object:
      obj: T

proc `=destroy`*[T](aref: Atomic[T]) =
  when T is ref:
    if aref.rp != nil:
      var cell = head(cast[pointer](aref.rp))
      echo "destroy aref: ", cast[pointer](aref.rp).repr, " rc: ", cell.rc, " cnt: ", cell.count()
      # `atomicDec` returns the new value
      if atomicDec(cell.rc, rcIncrement) == -rcIncrement:
        echo "\nlast <<<"
        inc cell.rc, rcIncrement # hack to re-use normal destroy
        `=destroy`(aref.rp)
        echo ">> done"
  elif T is object:
    # `=destroy`(aref)
    discard

proc `=copy`*[T](dest: var Atomic[T]; source: Atomic[T]) =
  echo "copy"
  when T is ref:
    # protect against self-assignments:
    if dest.rp != source.rp:
      `=destroy`(dest)
      wasMoved(dest)
      dest.rp = source.rp
      var cell = head(cast[pointer](dest.rp))
      discard atomicInc(cell.rc, rcIncrement)
      echo "copy cnt: ", cell.count
  elif T is object:
    # `=destroy`(aref)
    discard

proc newAtomic*[T: ref](obj: sink T): Atomic[T] =
  result = Atomic[T](rp: move obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc newAtomic*[T: object](obj: T): Atomic[T] =
  result = Atomic[T](obj: obj)

proc newAtomicRef*[T: ref](obj: T): Atomic[T] =
  result = Atomic[T](rp: obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc unsafeGet*[T](aref: Atomic[T]): lent T =
  when T is ref:
    aref.rp
  elif T is object:
    aref.obj

proc unsafeCount*[T](aref: ref T): int =
  var cell = head(cast[pointer](aref))
  cell.count()


import macros
import macrocache
import typetraits

const mcTable = CacheTable"subTest"

macro atomicAccessors*(tp: typed) =

  echo "TP: ", tp.treeRepr

  var timpl, tname: NimNode
  if tp.kind == nnkSym:
    timpl = tp.getImpl()
    timpl.expectKind(nnkTypeDef)
    tname = tp
  elif tp.kind == nnkRefTy:
    timpl = tp[^1].getImpl()
    tname = tp

  echo "TIMPL: ", timpl.treeRepr

  var tbody = timpl[^1]
  if tbody.kind == nnkRefTy:
    tbody = tbody[0]

  if tbody.kind == nnkSym:
    let ity = tbody.getImpl()
    ity.expectKind(nnkTypeDef)
    tbody = ity[^1]
  tbody.expectKind(nnkObjectTy)

  let idents = tbody[^1]
  idents.expectKind(nnkRecList)
  result = newStmtList()
  for ident in idents:
    if ident[0].kind != nnkPostFix:
      echo "TIDENT: cont"
      continue
    let name = ident(ident[0][1].strVal)
    let fieldName = ident ident[0][1].repr
    let fieldTp = ident[1]
    let obj = ident "obj"
    let fieldKd = fieldTp.getType()
    let fieldIsRef = fieldKd.kind == nnkBracketExpr and fieldKd[0].strVal == "ref"
    let fieldIsObj = fieldKd.kind == nnkObjectTy
    let fieldKey = fieldName.repr & "::" & fieldTp.repr

    if fieldKey in mcTable: continue
    else: mcTable[fieldKey] = fieldTp

    if fieldIsRef:
      echo "TP:REF: "
      result.add quote do:
        proc `name`*(`obj`: Atomic[`tname`]): Atomic[`fieldTp`] =
          newAtomicRef(`obj`.unsafeGet().`fieldName`)
        atomicAccessors(`fieldTp`)
    elif fieldIsObj:
      echo "TP:OBJ: "
      result.add quote do:
        proc `name`*(`obj`: Atomic[`tname`]): Atomic[`fieldTp`] =
          newAtomic(`obj`.unsafeGet().`fieldName`)
        atomicAccessors(`fieldTp`)
    else:
      echo "TP:ELSE: "
      result.add quote do:
        proc `name`*(`obj`: Atomic[`tname`]): `fieldTp` =
          `obj`.unsafeGet().`fieldName`


  echo "RES:\n", result.repr

when isMainModule:
  type
    Test* = ref object
      msg*: string
    Test2* = ref object
      msg*: string

    Bar* = object
      field*: Test

    Foo* = ref object
      inner*: Test
      outer*: Bar

  atomicAccessors(Foo)

  # type
  #   Test2* = object
  #     msg2*: string
  #   TestRef* = ref Test2

  #   Foo2* = object
  #     inner2*: ref Test2
  #   FooRef* = ref Foo2
  # # expandMacros:
  # # atomicAccessors(FooRef)

  # atomicAccessors(ref Foo2)
