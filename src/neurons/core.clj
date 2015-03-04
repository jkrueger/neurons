(ns neurons.core
  (:require
    [neurons.layer  :as layer]
    [incanter.core  :as m]
    [incanter.stats :as s]))

(defn make [& layers]
  {:layers (vec layers)})

(defn- forward [net input]
  (loop [layers      (:layers net)
         a           input
         activations [input]]
    (if-let [layer (first layers)]
      (let [a (layer/forward layer a)]
        (recur (rest layers)
               (:activation a)
               (conj activations a)))
      activations)))

(def run (comp :activation peek forward))

(defn backpropagate
  [net in target ^double rate]
  (loop [activations (forward net in)
         delta       (layer/error (peek activations) target)
         layers      '()]
    (if-let [a (peek activations)]
      (let [layer (layer/learn a delta rate)
            delta (layer/backward a delta)]
        (recur (pop activations)
               delta
               (cons a layers)))
      layers)))

(defn- label->target [label]
  (m/matrix
    (assoc (vec (repeat 10 0.0))
           (int label)
           1.0)
    1))

(defn- update-network [rate net batch]
  (let [samples (count batch)
        rate    (/ rate samples)]
    (reduce
      (fn [net [data target]]
        (let [layers (backpropagate net data target rate)]
          (assoc net :layers layers)))
      net
      batch)))

(defn learn [net data epochs batch-size rate]
  (let [data  (map (fn [[x y]] (vector x (label->target y))) data)
        epoch (atom 0)]
    (nth (iterate
          (fn [net]
            (println "starting epoch" (swap! epoch inc))
            (->> (shuffle data)
                 (partition batch-size)
                 (reduce (partial update-network rate) net)))
          net)
         epochs)))

(defn confidences [x]
  (->> x m/to-vect (map-indexed #(vector %2 %1)) sort reverse))

(defn guess [x]
  (second (first (confidences x))))

(defn accuracy [net data]
  (->> data
       (filter
         (fn [[sample target]]
           (= (guess (run net sample))
              target)))
       (count)
       (* (/ 1 (count data))) (float)))
