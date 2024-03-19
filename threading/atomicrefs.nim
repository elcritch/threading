
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
  echo "destroy aref: ", cast[pointer](aref.rp).repr
  if aref.rp != nil:
    var cell = head(cast[pointer](aref.rp))
    echo "decl aref: ", cell.rc, " ", cell.count()
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


