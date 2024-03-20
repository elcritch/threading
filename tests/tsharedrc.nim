
import threading/sharedrc


type
  Test* = object
    msg*: string

  Bar* = object
    field*: Test

  Foo* = ref object
    inner*: ref Test
    other*: Bar

proc `=destroy`*(obj: Test) =
  echo "destroying Test obj: ", obj.msg
  `=destroy`(obj.msg)

proc `=destroy`*(obj: Bar) =
  echo "destroying Bar obj: ", obj.field
  `=destroy`(obj.field)


atomicAccessors(Foo)


proc testDeep() =

  let test = Test.new()
  test.msg = "hello world!"
  var t1 = newSharedRc(Foo(inner: test))
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
    let y: SharedRc[ref Test] = t1.inner
    echo "y: addr: ", cast[pointer](y.unsafeGet).repr
    echo "t1:inner:count: ", t1.unsafeGet.inner.unsafeCount()
    echo "y: ", y.msg, "y:count: ", y.unsafeGet.unsafeCount()

  echo ""
  echo "t1:inner:count: ", t1.unsafeGet.inner.unsafeCount()
  echo "t2.inner.post:", " isUnique: ", t2.inner.unsafeGet.isUniqueRef

testDeep()

# proc testThread() =
#   proc test(aref: SharedRc[Test]) {.thread.} =
#     var lref = aref
#     echo "thread: ", lref[].msg
#   var thread: Thread[SharedRc[Test]]
#   var t1 = newSharedRc[Test](Test(msg: "hello world!"))
#   var t2 = t1
#   createThread(thread, test, t1)
#   thread.joinThread()
#   echo "t2: ", t2[].msg
# testThread()
# echo "done"

when false:
  when string is ref:
    proc msg*(obj: SharedRc[Test]): SharedRc[string] =
      newSharedRcRef(obj.unsafeGet().msg)
    atomicAccessors(string)
  else:
    proc msg*(obj: SharedRc[Test]): string =
      obj.unsafeGet().msg

  when ref Test2 is ref:
    proc inner2*(obj: SharedRc[ref Foo2]): SharedRc[ref Test2] =
      newSharedRcRef(obj.unsafeGet().inner2)
    atomicAccessors(ref Test2)
  else:
    proc inner2*(obj: SharedRc[ref Foo2]): ref Test2 =
      obj.unsafeGet().inner2