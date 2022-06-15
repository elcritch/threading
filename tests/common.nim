import std/[os, logging, unittest, strutils, strformat]
export os, logging, unittest, strutils, strformat

import threading/channels

const
  testLogLevel {.strdefine.}: string = "lvlInfo"
  logLevel*: Level = parseEnum[Level](testLogLevel)

var logger* = newConsoleLogger(logLevel, verboseFmtStr)

template runMultithreadInOrderTest*[T](chan: Chan[T]) =
  var worker1: Thread[void]

  # Launch the first worker.
  createThread(worker1) do:
    chan.send("Hello World!")

  # Wait for the thread to exit before moving on to the next example.
  worker1.joinThread()

  var dest = ""
  chan.recv(dest)
  logger.log(lvlDebug, "Received msg: " & $dest)
  doAssert dest == "Hello World!"


template runMultithreadBockTest*[T](chan: Chan[T]) =
  var worker2: Thread[void]
  # Launch the other worker.
  createThread(worker2) do: 
    # This is another proc to run in a background thread. This proc takes a while
    # to send the message since it sleeps for 2 seconds (or 2000 milliseconds).
    sleep(2000)
    chan.send("Another message")

  # This time, use a non-blocking approach with tryRecv.
  # Since the main thread is not blocked, it could be used to perform other
  # useful work while it waits for data to arrive on the channel.

  var messages: seq[string]
  var msg = ""
  while true:
    let tried = chan.tryRecv(msg)
    if tried:
      logger.log(lvlDebug, "Receive msg: " & $msg)
      messages.add move(msg)
      break

    messages.add "Pretend I'm doing useful work..."
    logger.log(lvlDebug, "add fakeMsg to messages (other work): " & $messages.len())
    sleep(400)

  # Wait for the second thread to exit before cleaning up the channel.
  worker2.joinThread()

  # Clean up the channel.
  check messages[^1] == "Another message"
  check messages.len >= 2
