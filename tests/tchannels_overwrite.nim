discard """
  matrix: "--threads:on --gc:orc; --threads:on --gc:arc"
  disabled: "freebsd"
"""

import threading/channels
import std/[os, logging, unittest]

suite "testing Chan with overwrite mode":
  var logger = newConsoleLogger(levelThreshold=lvlInfo)
  
  setup:
    discard "run before each test"
  
  teardown:
    discard "run after each test"

  test "basic overwrite tests":
    # give up and stop if this fails
    require(true)
    var chan = newChan[string](elements = 4, overwrite = true)

    # This proc will be run in another thread using the threads module.
    for i in 1..10:
      logger.log(lvlDebug, "adding more messages than fit: " & $i)
      chan.send("msg" & $i)

    var messages: seq[string]
    var msg = ""
    while true:
      if chan.tryRecv(msg):
        messages.add move(msg)
      else:
        break

    logger.log(lvlDebug, "got messages: " & $messages)
    check messages == @["msg7", "msg8", "msg9", "msg10"]

  test "basic overwrite tests":

    var chan = newChan[string](elements = 4, overwrite = true)

    # Launch the worker.
    var worker1: Thread[void]
    var worker2: Thread[void]

    ## start worker1
    createThread(worker1, proc () =
      chan.send("Hello World!"))

    # Block until the message arrives, then print it out.
    block:
      var dest = ""
      chan.recv(dest)
      doAssert dest == "Hello World!"

      # Wait for the thread to exit before moving on to the next example.
      worker1.joinThread()

    # Launch the other worker.
    createThread(worker2, proc () =
      # This is another proc to run in a background thread. This proc takes a while
      # to send the message since it sleeps for 2 seconds (or 2000 milliseconds).
      sleep(2000)
      chan.send("Another message"))

    # This time, use a non-blocking approach with tryRecv.
    # Since the main thread is not blocked, it could be used to perform other
    # useful work while it waits for data to arrive on the channel.
    var messages: seq[string]
    block:
      var msg = ""
      while true:
        let tried = chan.tryRecv(msg)
        if tried:
          messages.add move(msg)
          break

        messages.add "Pretend I'm doing useful work..."
        # For this example, sleep in order not to flood stdout with the above
        # message.
        sleep(400)

      # Wait for the second thread to exit before cleaning up the channel.
      worker2.joinThread()

      # Clean up the channel.
      doAssert messages[^1] == "Another message"
      doAssert messages.len >= 2


