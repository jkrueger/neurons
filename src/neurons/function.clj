(ns neurons.function)

(defprotocol LearnableFunction
  (draw [_ n m])
  (sample [_]))

(defrecord SampledFunction
  [samples]
  LearnableFunction
  (draw [_ n m]
    (->> (repeatedly #(shuffle samples))
         (apply concat)
         (take (* n m))
         (partition m )))
  (sample [_]
    (rand-nth samples)))

(defn sampled [samples]
  (SampledFunction. samples))
