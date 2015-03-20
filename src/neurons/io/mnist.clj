(ns neurons.io.mnist
  (:refer-clojure :exclude (load))
  (:require
    [clojure.java.io :as io]
    [incanter.core :as m]
    [neurons.function :as function]))

(def training-set
  ["data/train-images.idx3-ubyte" "data/train-labels.idx1-ubyte"])

(def verification-set
  ["data/t10k-images.idx3-ubyte" "data/t10k-labels.idx1-ubyte"])

(def width  28)
(def height 28)

(def sample-size   (* width height))
(def data-set-size 60000)

(defn byte->gradient [b]
  (double (/ (bit-and (int b) 0xff) 255)))

(defn- label->target [label]
  (m/matrix
    (assoc (vec (repeat 10 0.0))
           (int label)
           1.0)
    1))

(defn load [[data labels]]
  (let [in-data  (io/input-stream data)
        in-label (io/input-stream labels)
        sample   (byte-array sample-size)]
    (.skip in-data 16)
    (.skip in-label 8)
    (loop [samples []]
      (if (not= -1 (.read in-data sample))
        (let [label (.read in-label)
              mat   (m/matrix (map byte->gradient sample) 1)]
          (when (= (mod (count samples) 1000) 0)
            (println (+ (count samples) 1000)  "loaded"))
          (recur (conj samples
                       {:in     mat
                        :result (label->target label)
                        :label  label})))
        samples))))

(defn as-fun
  [data]
  (function/sampled data))
