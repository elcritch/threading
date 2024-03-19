
import threading/atomicrefs


type
  Test* = ref object
    msg*: string

  Foo* = ref object
    inner*: Test

proc `=destroy`*(obj: type(Test()[])) =
  echo "destroying Test obj: ", obj.msg
  `=destroy`(obj.msg)

proc inner*(obj: Atomic[Foo]): Atomic[Test] =
  newAtomicRef(obj.unsafeGet().inner)

proc msg*(obj: Atomic[Test]): string =
  obj.unsafeGet().msg


proc testDeep() =

  var t1 = newAtomic(Foo(inner: Test(msg: "hello world!")))
  var t2 = t1

  echo "t1: addr: ", cast[pointer](t1.unsafeGet).repr
  echo "t2: addr: ", cast[pointer](t2.unsafeGet).repr
  # echo "t2: ", head(cast[pointer](t2.unsafeGet)).count()

  echo "t1: ", t1.inner.msg
  echo "t2: ", t2.inner.msg
  echo "t2.inner:", " isUnique: ", t2.inner.unsafeGet.isUniqueRef
  let y: Atomic[Test] = t1.inner
  echo "y: addr: ", cast[pointer](y.unsafeGet).repr
  echo "y: ", y.msg, " isUnique: ", y.unsafeGet.isUniqueRef()

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
