import threading/channels
import std/unittest

const
  BufferSize = 3

suite "Ring Buffer Channel Tests":
  test "Non-blocking ring buffer behavior":
    
    proc fillBuffer(n: int): seq[int] =
      echo "\nfillBuffer ", n
      var chan = newChan[int](BufferSize, overwrite = true)
      # Fill the buffer
      for i in 0..<BufferSize+n:
        check chan.trySend(i)
      
      # Receive values - should get BufferSize as first value
      var values: seq[int]
      for i in 0..<BufferSize:
        var x: int
        if not chan.tryRecv(x):
          break
        values.add(x)
      
      # Verify we got the most recent values
      result = values
    
    check fillBuffer(0) == @[0, 1, 2]
    check fillBuffer(1) == @[1, 2, 3]
    check fillBuffer(2) == @[2, 3, 4]
    check fillBuffer(3) == @[3, 4, 5]
    check fillBuffer(4) == @[4, 5, 6]
