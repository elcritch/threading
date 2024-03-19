
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

  Test* = ref object
    msg*: string

  Foo* = ref object
    inner*: Test

proc `=destroy`*(obj: var type(Test()[])) =
  echo "destroying Test obj: ", obj.msg
  `=destroy`(obj.msg)

proc `=destroy`*[T](aref: var Atomic[T]) =
  echo "destroy aref: ", cast[pointer](aref.rp).repr
  if aref.rp != nil:
    var cell = head(cast[pointer](aref.rp))
    echo "decl aref: ", cell.rc, " ", cell.count()
    # `atomicDec` returns the new value
    if atomicDec(cell.rc, rcIncrement) == -rcIncrement:
      echo "\nlast <<<"
      inc cell.rc, rcIncrement # hack to re-use normal destroy
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
  result = Atomic[T](rp: move obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc newAtomicRef[T: ref](obj: T): Atomic[T] =
  result = Atomic[T](rp: obj)
  var cell = head(cast[pointer](result.rp))
  discard atomicInc(cell.rc, rcIncrement)

proc inner*(obj: Atomic[Foo]): Atomic[Test] =
  newAtomicRef(obj.rp.inner)

proc msg*(obj: Atomic[Test]): string =
  obj.rp.msg


proc testDeep() =

  var t1 = newAtomic(Foo(inner: Test(msg: "hello world!")))
  var t2 = t1

  echo "t1: addr: ", cast[pointer](t1.rp).repr
  echo "t2: addr: ", cast[pointer](t2.rp).repr
  echo "t2: ", head(cast[pointer](t2.rp)).count()

  echo "t1: ", t1.inner.msg
  echo "t2: ", t2.inner.msg
  echo "t2.inner:", " isUnique: ", t2.inner.rp.isUniqueRef
  let y: Atomic[Test] = t1.inner
  echo "y: addr: ", cast[pointer](y.rp).repr
  echo "y: ", y.msg, " isUnique: ", y.rp.isUniqueRef()

testDeep()

# proc testThread() =
#   proc test(aref: Atomic[Test]) {.thread.} =
#     var lref = aref
#     echo "thread: ", lref[].msg
#   var thread: Thread[Atomic[Test]]
#   var t1 = newAtomic[Test](Test(msg: "hello world!"))
#   var t2 = t1
#   createThread(thread, test, t1)
#   thread.joinThread()
#   echo "t2: ", t2[].msg
# testThread()
# echo "done"
