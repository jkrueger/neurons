(ns neurons.io.mnist
  (:refer-clojure :exclude (load))
  (:require
    [clojure.java.io :as io]
    [incanter.core :as m]))

(def width  28)
(def height 28)

(def sample-size (* width height))

(defn byte->gradient [b]
  (double (/ (bit-and (int b) 0xff) 255)))

(defn load [data labels]
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
            (println (count samples)  "loaded"))
          (recur (conj samples [mat label])))
        samples))))
