package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
)

func loadData() ([][]float64, []string) {
	f, err := os.Open("iris.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','
	r.LazyQuotes = true
	_, err = r.Read()
	if err != nil {
		panic(err)
	}
	rows, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	X := [][]float64{}
	Y := []string{}

	for _, cols := range rows {
		x := make([]float64, 4)
		y := cols[4]
		for j, s := range cols[:4] {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				panic(err)
			}
			x[j] = v
		}
		X = append(X, x)
		Y = append(Y, y)
	}
	return X, Y
}

type KNN struct {
	k  int
	XX [][]float64
	Y  []string
}

func distance(lhs, rhs []float64) float64 {
	val := 0.0
	for i, _ := range lhs {
		val += math.Pow(lhs[i]-rhs[i], 2)
	}
	return math.Sqrt(val)
}

func (knn *KNN) predict(X [][]float64) []string {
	results := []string{}
	for _, x := range X {
		type item struct {
			i int
			f float64
		}
		var items []item
		for i, xx := range knn.XX {
			items = append(items, item{
				i: i,
				f: distance(x, xx),
			})
		}
		sort.Slice(items, func(i, j int) bool {
			return items[i].f < items[j].f
		})

		var labels []string
		for i := 0; i < knn.k; i++ {
			labels = append(labels, knn.Y[items[i].i])
		}

		founds := map[string]int{}
		for _, label := range labels {
			founds[label] += 1
		}

		type rank struct {
			i int
			s string
		}
		var ranks []rank
		for k, v := range founds {
			ranks = append(ranks, rank{
				i: v,
				s: k,
			})
		}
		sort.Slice(ranks, func(i, j int) bool {
			return ranks[i].i > ranks[j].i
		})
		results = append(results, ranks[0].s)
	}
	return results
}

func main() {
	X, Y := loadData()
	var trainX, testX [][]float64
	var trainY, testY []string
	for i, _ := range X {
		if i%2 == 0 {
			trainX = append(trainX, X[i])
			trainY = append(trainY, Y[i])
		} else {
			testX = append(testX, X[i])
			testY = append(testY, Y[i])
		}
	}

	knn := KNN{
		k:  8,
		XX: trainX,
		Y:  trainY,
	}

	predicted := knn.predict(testX)
	correct := 0
	for i, _ := range predicted {
		if predicted[i] == testY[i] {
			correct += 1
		}
	}

	fmt.Printf("%f%%\n", float64(correct)/float64(len(predicted))*100)
}
