
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
    # rp: T
    rp {.cursor.}: T

  Test* = ref object
    msg*: string

proc `=destroy`*(obj: var type(Test()[])) =
  echo "destroying Test obj ", obj.msg
  `=destroy`(obj.msg)

proc `=destroy`*[T](aref: var Atomic[T]) =
  echo "destroy aref: ", cast[pointer](aref.rp).repr
  if aref.rp != nil:
    var cell = head(cast[pointer](aref.rp))
    echo "decl aref: ", cell.rc, " ", cell.count()
    # `atomicDec` returns the new value
    if atomicDec(cell.rc, rcIncrement) == 0:
      echo "\nlast <<<"
      `=destroy`(aref.rp)
      aref.rp = nil
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
  result = Atomic[Test](rp: move obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, 2*rcIncrement)

proc `[]`*[T: ref object](aref: Atomic[T]): lent T =
  aref.rp

# proc testBasic() =
#   proc test(aref: Atomic[Test]) {.thread.} =
#     var lref = aref
#     echo "thread: ", lref[].msg

#   var thread: Thread[Atomic[Test]]
#   var t1 = newAtomic[Test](Test(msg: "hello world!"))
#   var t2 = t1

#   createThread(thread, test, t1)
#   thread.joinThread()
#   echo "t2: ", t2[].msg

# testBasic()
# echo "done"

proc testBasic() =

  var t1 = newAtomic[Test](Test(msg: "hello world!"))
  var t2 = t1

  echo "t1: ", cast[pointer](t1.rp).repr
  echo "t2: ", cast[pointer](t2.rp).repr
  echo "t2: ", head(cast[pointer](t2.rp)).count()

  echo "t1: ", t1[].msg
  echo "t2: ", t2[].msg

testBasic()
echo "done"