//! In-process loopback transport used by tests, examples, and FSM mocks.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{DdsMessage, Transport, TransportError};

/// Reader handle returned by [`Transport::subscribe`].
///
/// Each clone of the underlying [`InMemoryTransport`] shares the same set of
/// queues, so a publish in one clone is visible to a subscription in any
/// other clone.
pub struct Subscription<T> {
    inner: Arc<Mutex<TopicQueue>>,
    _ty: std::marker::PhantomData<fn() -> T>,
}

impl<T> Subscription<T> {
    /// Returns how many byte-buffers are currently queued.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.inner.lock().map(|g| g.queue.len()).unwrap_or_default()
    }
}

impl<T> std::fmt::Debug for Subscription<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription")
            .field("pending", &self.pending())
            .finish()
    }
}

impl<T: DdsMessage + 'static> Subscription<T> {
    /// Returns the next decoded message if one is buffered, else `None`.
    #[must_use]
    pub fn try_recv(&self) -> Option<T> {
        let bytes = self.inner.lock().ok()?.queue.pop_front()?;
        // Best-effort decode; on failure we drop the message and return None.
        T::decode(&bytes).ok()
    }

    /// Blocks (poll-loop) up to `timeout` waiting for a message.
    #[must_use]
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(msg) = self.try_recv() {
                return Some(msg);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

#[derive(Default)]
struct TopicQueue {
    queue: std::collections::VecDeque<Vec<u8>>,
    type_tag: Option<&'static str>,
}

/// Loopback transport: every `publish(topic, msg)` is routed to all
/// in-process subscribers of the same topic.
///
/// Cloning is cheap and shares the same internal channels — useful for
/// "publisher half" / "subscriber half" patterns in tests.
#[derive(Default, Clone)]
pub struct InMemoryTransport {
    topics: Arc<Mutex<HashMap<String, Arc<Mutex<TopicQueue>>>>>,
}

impl InMemoryTransport {
    /// Create a fresh, empty transport.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn queue_for(&self, topic: &str) -> Arc<Mutex<TopicQueue>> {
        let mut topics = self
            .topics
            .lock()
            .expect("InMemoryTransport poisoned mutex");
        topics.entry(topic.to_owned()).or_default().clone()
    }
}

impl std::fmt::Debug for InMemoryTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let topics = self
            .topics
            .lock()
            .map(|g| g.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        f.debug_struct("InMemoryTransport")
            .field("topics", &topics)
            .finish()
    }
}

impl Transport for InMemoryTransport {
    fn publish<T: DdsMessage>(&mut self, topic: &str, msg: &T) -> Result<(), TransportError> {
        let bytes = msg.encode()?;
        let queue = self.queue_for(topic);
        let mut guard = queue.lock().map_err(|_| TransportError::Closed)?;
        // First publish on a topic stamps the type tag; later publishes must match.
        match guard.type_tag {
            None => guard.type_tag = Some(T::type_name()),
            Some(existing) if existing != T::type_name() => {
                return Err(TransportError::Publish {
                    topic: topic.to_owned(),
                    reason: format!(
                        "topic type mismatch: existing {existing}, attempted {}",
                        T::type_name()
                    ),
                });
            }
            _ => {}
        }
        guard.queue.push_back(bytes);
        Ok(())
    }

    fn subscribe<T: DdsMessage + 'static>(
        &mut self,
        topic: &str,
    ) -> Result<Subscription<T>, TransportError> {
        let queue = self.queue_for(topic);
        // Touch the type tag so subscribe-before-publish still pins it.
        {
            let mut guard = queue.lock().map_err(|_| TransportError::Closed)?;
            match guard.type_tag {
                None => guard.type_tag = Some(T::type_name()),
                Some(existing) if existing != T::type_name() => {
                    return Err(TransportError::Subscribe {
                        topic: topic.to_owned(),
                        reason: format!(
                            "topic type mismatch: existing {existing}, requested {}",
                            T::type_name()
                        ),
                    });
                }
                _ => {}
            }
        }
        Ok(Subscription {
            inner: queue,
            _ty: std::marker::PhantomData,
        })
    }
}

// `Any` import is only used through generic bounds in tests; suppress unused.
#[allow(dead_code)]
fn _ensure_any_imported(_: &dyn Any) {}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use unitree_hg_idl::{LowCmd, LowState};

    #[test]
    fn publish_then_subscribe_delivers_message() {
        let mut tx = InMemoryTransport::new();
        let mut cmd = LowCmd::default();
        cmd.mode_pr = 7;
        let sub = tx.subscribe::<LowCmd>("rt/lowcmd").expect("subscribe");
        tx.publish("rt/lowcmd", &cmd).expect("publish");

        let received = sub.try_recv().expect("message must be queued");
        assert_eq!(received.mode_pr, 7);
        assert!(received.verify_crc(), "CRC must round-trip");
    }

    #[test]
    fn publish_before_subscribe_is_lost_after_drain() {
        // The in-memory transport keeps a buffered queue; subscribing late is
        // still served from the buffer.
        let mut tx = InMemoryTransport::new();
        let cmd = LowCmd::default();
        tx.publish("rt/lowcmd", &cmd).expect("publish");

        let sub = tx.subscribe::<LowCmd>("rt/lowcmd").expect("subscribe");
        assert!(sub.try_recv().is_some());
        assert!(sub.try_recv().is_none(), "queue should now be empty");
    }

    #[test]
    fn type_mismatch_on_topic_is_rejected() {
        let mut tx = InMemoryTransport::new();
        let _sub = tx
            .subscribe::<LowCmd>("rt/lowcmd")
            .expect("subscribe LowCmd");
        let state = LowState::default();
        let err = tx
            .publish("rt/lowcmd", &state)
            .expect_err("type mismatch must be rejected");
        match err {
            TransportError::Publish { topic, .. } => assert_eq!(topic, "rt/lowcmd"),
            other => panic!("expected Publish error, got {other:?}"),
        }
    }

    #[test]
    fn shared_clones_share_queues() {
        let mut tx = InMemoryTransport::new();
        let mut tx2 = tx.clone();
        let sub = tx.subscribe::<LowCmd>("rt/lowcmd").expect("subscribe");
        tx2.publish("rt/lowcmd", &LowCmd::default())
            .expect("publish on clone");
        assert!(sub.try_recv().is_some(), "clone-published msg must arrive");
    }

    #[test]
    fn recv_timeout_returns_none_when_idle() {
        let mut tx = InMemoryTransport::new();
        let sub = tx.subscribe::<LowCmd>("rt/lowcmd").expect("subscribe");
        let start = Instant::now();
        let got = sub.recv_timeout(Duration::from_millis(20));
        assert!(got.is_none());
        assert!(start.elapsed() >= Duration::from_millis(20));
    }
}
