(ns neurons.layer)

(defprotocol Layer
  (forward [_ in]))

(defprotocol Activation
  (error [_ target])
  (backward [_ delta weights])
  (learn [_ delta rate]))

(defmulti make :type)
