using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNN
{
    internal class RecurrentNeuralNetwork
    {
        public List<double> Hidden_states { get; set; }
        public List<Neuron> Neurons { get; set; }
        public double Actual { get; set; }
        public double Predicted { get; set; }
        public RecurrentNeuralNetwork()
        {
            Hidden_states = new List<double>();
            Neurons = new List<Neuron>();
        }
        public void Initialize(List<string> input_values)
        {
            List<double> values = new List<double>();
            foreach (string value in input_values)
            {
                values.Add( double.Parse(value, CultureInfo.InvariantCulture) );
            }

            Actual = values.Last();

            for (int i = 0; i < values.Count; i++)
            {
                this.Neurons.Add(new Neuron());
                this.Neurons[i].Initialize(values[i]);
            }

        }
        public void CalculateActivations()
        {
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                Hidden_states.Add( Neurons[i].CalculateActivation() + SumUntilIndex(i) );
            }

            Predicted = Neurons.Last().Activation;
        }
        public double Error()
        {
            return 0.5 * Math.Pow(Predicted - Actual, 2);
        }
        public double ErrorDerivative()
        {
            return Predicted - Actual;
        }

        private double SumUntilIndex(int index)
        {
            double sum = 0;
            int localIndex = 0;
            while (localIndex < index)
            {
                sum += Hidden_states[localIndex];
                localIndex++;
            }
            return sum;
        }
    }
}
