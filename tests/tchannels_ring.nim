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
    check values == @[BufferSize, 1, 2]
  
  test "Non-blocking ring buffer behavior with 2 elements":
    var chan = newChan[int](BufferSize, overwrite = true)
    
    # Fill the buffer
    for i in 0..<BufferSize+1:
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
    check values == @[BufferSize, BufferSize+1, 2]

  test "Multiple overwrites":
    var chan = newChan[int](BufferSize, overwrite = true)
    
    # Fill the buffer
    for i in 0..<BufferSize:
      discard chan.trySend(i)
    
    # Send multiple values that should overwrite
    for i in 0..<BufferSize:
      discard chan.trySend(BufferSize + i)
    
    # Receive values - should get the most recent values
    var values: seq[int]
    for i in 0..<BufferSize:
      var x: int
      let res = chan.tryRecv(x)
      if not res:
        break
      values.add(x)
    
    # Verify we got the most recent values
    check values == @[BufferSize + 1, BufferSize + 2, BufferSize + 3]
  