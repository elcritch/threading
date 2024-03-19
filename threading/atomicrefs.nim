

type
  Atomic*[T: ref object] = object
    obj {.cursor.}: T

  Test* = ref object
    msg*: string

block trySend_recv:
  proc test(aref: Atomic[Test]) {.thread.} =
    echo "thread: ", aref

  var thread: Thread[Atomic[Test]]
  var t1 = Atomic[Test](obj: Test(msg: "hello world!"))

  createThread(thread, test, t1)

  thread.joinThread()
