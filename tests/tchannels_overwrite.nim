discard """
  matrix: "--threads:on --gc:orc; --threads:on --gc:arc"
  disabled: "freebsd"
"""

import threading/channels
import std/logging

var logger = newConsoleLogger(levelThreshold=lvlInfo)

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

doAssert messages == @["msg7", "msg8", "msg9", "msg10"]

# doAssert dest == "Hello World!"

