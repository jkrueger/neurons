(ns neurons.layer.mlp
  (:require
    [neurons.layer  :as layer]
    [incanter.core  :as m]
    [incanter.stats :as s]))

(defn- logistic-activation [^double x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn- activate [v f]
  (m/matrix (m/matrix-map f v) 1))

(defn- deactivate [v f]
  (let [m (m/matrix-map
           (fn [^double x]
             (let [s (f x)]
               (* s (- 1 s))))
           v)]
    (m/matrix m 1)))

(defn- cost-derivative [mat y]
  (m/minus mat y))

(defn- improve [x delta ^double rate]
  (m/minus x (m/mmult delta rate)))

(defrecord Activation [layer in z activation]

  layer/Activation

  (error [_ target]
    (let [delta-a (cost-derivative activation target)]
      (m/mult delta-a (deactivate z (:activation-fn layer)))))

  (backward [_ delta weights]
    (let [spv (deactivate z (:activation-fn layer))]
      (-> (m/mmult (m/trans weights) delta)
          (m/mult spv))))

  (learn [_ delta rate]
    (let [delta-w (m/mmult delta (m/trans in))]
      (assoc layer
        :weights (improve (:weights layer) delta-w rate)
        :biases  (improve (:biases layer) delta rate)))))

(defrecord Layer [size weights biases activation-fn]

  layer/Layer

  (forward [this in]
    (let [z (m/plus (m/mmult weights in) biases)]
      (Activation. this in z (activate z activation-fn)))))

(defn- make-biases [size]
  (m/matrix (s/sample-normal size) 1))

(defn- make-weights [size in]
  (let [incoming (:size in)
        weights  (* incoming size)]
    (m/matrix (s/sample-normal weights) incoming)))

(defmulti make-activation-fn (fn [k] k))

(defmethod make-activation-fn :logistic [_]
  logistic-activation)

(defn make [size in & {:keys [activation] :or {:activation logistic-activation}}]
  (map->Layer
    {:size          size
     :weights       (make-weights size in)
     :biases        (make-biases size)
     :activation-fn (make-activation-fn activation)}))
