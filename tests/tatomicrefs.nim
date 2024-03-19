
import threading/atomicrefs


type
  Test* = object
    msg*: string

  Foo* = ref object
    inner*: ref Test

proc `=destroy`*(obj: Test) =
  echo "destroying Test obj: ", obj.msg
  `=destroy`(obj.msg)

atomicAccessors(Foo)
when false:
  when string is ref:
    proc msg*(obj: Atomic[Test]): Atomic[string] =
      newAtomicRef(obj.unsafeGet().msg)
    atomicAccessors(string)
  else:
    proc msg*(obj: Atomic[Test]): string =
      obj.unsafeGet().msg

  when ref Test2 is ref:
    proc inner2*(obj: Atomic[ref Foo2]): Atomic[ref Test2] =
      newAtomicRef(obj.unsafeGet().inner2)
    atomicAccessors(ref Test2)
  else:
    proc inner2*(obj: Atomic[ref Foo2]): ref Test2 =
      obj.unsafeGet().inner2

proc testDeep() =

  let test = Test.new()
  test.msg = "hello world!"
  var t1 = newAtomic(Foo(inner: test))
  var t2 = t1

  echo "t1: addr: ", cast[pointer](t1.unsafeGet).repr
  echo "t2: addr: ", cast[pointer](t2.unsafeGet).repr
  # echo "t2: ", head(cast[pointer](t2.unsafeGet)).count()

  echo "t1:inner: ", cast[pointer](t1.inner.unsafeGet).repr
  echo "t2:inner: ", cast[pointer](t2.inner.unsafeGet).repr
  echo "t1:inner:count: ", t1.unsafeGet.inner.unsafeCount()
  echo "t2:inner:count: ", t2.unsafeGet.inner.unsafeCount()
  echo "t2:inner:count: ", t2.unsafeGet.inner.unsafeCount()

  block:
    echo ""
    let y: Atomic[ref Test] = t1.inner
    echo "y: addr: ", cast[pointer](y.unsafeGet).repr
    echo "t1:inner:count: ", t1.unsafeGet.inner.unsafeCount()
    echo "y: ", y.msg, "y:count: ", y.unsafeGet.unsafeCount()

  echo ""
  echo "t1:inner:count: ", t1.unsafeGet.inner.unsafeCount()
  echo "t2.inner.post:", " isUnique: ", t2.inner.unsafeGet.isUniqueRef

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
