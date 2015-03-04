(ns neurons.layer.sigmoid
  (:require
    [neurons.layer  :as layer]
    [incanter.core  :as m]
    [incanter.stats :as s]))

(defn- sigmoid* [^double x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn- sigmoid [v]
  (m/matrix (m/matrix-map sigmoid* v) 1))

(defn- sigmoid-prime [v]
  (let [m (m/matrix-map
           (fn [^double x]
             (let [s (sigmoid* x)]
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
      (m/mult delta-a (sigmoid-prime z))))

  (backward [_ delta weights]
    (let [spv (sigmoid-prime z)]
      (-> (m/mmult (m/trans weights) delta)
          (m/mult spv))))

  (learn [_ delta rate]
    (let [delta-w (m/mmult delta (m/trans in))]
      (assoc layer
        :weights (improve (:weights layer) delta-w rate)
        :biases  (improve (:biases layer) delta rate)))))

(defrecord Layer [size weights biases]

  layer/Layer

  (forward [this in]
    (let [z (m/plus (m/mmult weights in) biases)]
      (Activation. this in z (sigmoid z)))))

(defn- make-biases [size]
  (m/matrix (s/sample-normal size) 1))

(defn- make-weights [size in]
  (let [incoming (:size in)
        weights  (* incoming size)]
    (m/matrix (s/sample-normal weights) incoming)))

(defn make [size in]
  (map->Layer
    {:size    size
     :weights (make-weights size in)
     :biases  (make-biases size)}))
