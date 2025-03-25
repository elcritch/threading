import threading/channels
import std/unittest

const
  BufferSize = 3

suite "Ring Buffer Channel Tests":
  test "Non-blocking ring buffer behavior":
    var chan = newChan[int](BufferSize, overwrite = true)
    
    # Fill the buffer
    for i in 0..<BufferSize:
      check chan.trySend(i)
    
    # Send one more - should overwrite oldest value
    discard chan.trySend(BufferSize)
    
    # Receive values - should get BufferSize as first value
    var values: seq[int]
    for i in 0..<BufferSize:
      var x: int
      check chan.tryRecv(x)
      values.add(x)
    
    # Verify we got the most recent values
    check values == @[1, 2, BufferSize]

