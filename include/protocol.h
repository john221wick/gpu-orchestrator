#pragma once
// JSON protocol constants and message types for client-daemon communication.
// Transport: Unix domain socket at /tmp/gpu-scheduler.sock
// Format: newline-terminated JSON strings

// Client -> Daemon message types
#define PROTO_TYPE_SUBMIT   "submit"
#define PROTO_TYPE_STATUS   "status"
#define PROTO_TYPE_CANCEL   "cancel"

// Daemon -> Client response fields
#define PROTO_STATUS_ACCEPTED  "accepted"
#define PROTO_STATUS_ERROR     "error"

// Socket path
#define SCHEDULER_SOCKET_PATH  "/tmp/gpu-scheduler.sock"

// Max message size (bytes)
#define PROTO_MAX_MSG_SIZE  (64 * 1024)
