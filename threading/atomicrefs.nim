

type
  Atomic*[T: ref object] = object
    obj {.cursor.}: T

  Test* = ref object
    msg*: string

proc `[]`*[T: ref object](aref: Atomic[T]): lent T =
  aref.obj

proc testBasic() =
  proc test(aref: Atomic[Test]) {.thread.} =
    echo "thread: ", aref[].msg

  var thread: Thread[Atomic[Test]]
  var t1 = Atomic[Test](obj: Test(msg: "hello world!"))

  createThread(thread, test, t1)
  thread.joinThread()

testBasic()
