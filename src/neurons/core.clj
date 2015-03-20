(ns neurons.core
  (:require
    [neurons.layer    :as layer]
    [neurons.function :as function]
    [incanter.core    :as m]
    [incanter.stats   :as s]))

(defn make [net]
  (layer/make net))

(defn- forward [net input]
  (loop [layers      (:layers net)
         a           input
         activations []]
    (if-let [layer (first layers)]
      (let [a (layer/forward layer a)]
        (recur (rest layers)
               (:activation a)
               (conj activations a)))
      activations)))

(def run (comp :activation peek forward))

(defn backpropagate
  [net in target ^double rate]
  (let [activations (forward net in)]
    (loop [prev   (peek activations)
           as     (pop activations)
           delta  (layer/error prev target)
           layers (list (layer/learn prev delta rate))]
      (if-let [a (peek as)]
        (let [weights (get-in prev [:layer :weights])
              delta   (layer/backward a delta weights)
              updated (layer/learn a delta rate)]
          (recur a
                 (pop as)
                 delta
                 (cons updated layers)))
        layers))))



(defn- update-network [rate net batch]
  (let [samples (count batch)
        rate    (/ rate samples)]
    (reduce
      (fn [net sample]
        (let [data   (:in sample)
              target (:result sample)
              layers (backpropagate net data target rate)]
          (assoc net :layers layers)))
      net
      batch)))

(defn learn [net f epochs data-size batch-size rate]
  (let [sample-size (/ data-size batch-size)]
    (nth (iterate
          (fn [net]
            (reduce
              (partial update-network rate)
              net
              (function/draw f sample-size batch-size)))
          net)
         epochs)))

(defn confidences [x]
  (->> x m/to-vect (map-indexed #(vector %2 %1)) sort reverse))

(defn guess [x]
  (second (first (confidences x))))

(defn accuracy [net data]
  (->> data
       (filter
         (fn [sample]
           (= (guess (run net (:in sample)))
              (:label sample))))
       (count)
       (* (/ 1 (count data))) (float)))
