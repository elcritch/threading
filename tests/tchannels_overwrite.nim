discard """
  matrix: "--threads:on --gc:orc; --threads:on --gc:arc"
  disabled: "freebsd"
"""

import threading/channels
import std/[logging, unittest]

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


