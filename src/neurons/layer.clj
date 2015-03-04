(ns neurons.layer)

(defprotocol Layer
  (forward [_ in]))

(defprotocol Activation
  (error [_ target])
  (backward [_ delta])
  (learn [_ delta  ^double rate]))
