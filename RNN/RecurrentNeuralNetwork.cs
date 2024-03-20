using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace RNN
{
    public enum NeuronType
    {
        Recurrent,
        Output
    }
    internal class RecurrentNeuralNetwork
    {
        public readonly static double LearningRate = 0.0005;
        public List<double> Hidden_states { get; set; }
        public List<Neuron> Neurons { get; set; }
        public double Actual { get; set; }
        public double Predicted { get; set; }

        #region Private fields
        private double errorDerivative = 0;
        private double error = 0;
        #endregion // Private fields

        #region Contructor
        public RecurrentNeuralNetwork()
        {
            Hidden_states = new List<double>();
            Neurons = new List<Neuron>();
        }
        #endregion // Constructor


        public void Initialize(List<string> input_values)
        {
            List<double> values = new List<double>();
            Hidden_states = new List<double>();
            for (int i = 0; i < input_values.Count - 1; i++)
            {
                values.Add(double.Parse(input_values[i], CultureInfo.InvariantCulture) - 250); // normalizing
            }

            Actual = double.Parse( input_values.Last(), CultureInfo.InvariantCulture);

            if (Neurons.Count != 0)
            {
                for (int i = 0; i < values.Count - 1; i++)
                {
                    this.Neurons[i].Initialize(values[i]);
                }
            }
            else
            {
                for (int i = 0; i < values.Count - 1; i++)
                {
                    this.Neurons.Add(new Neuron(NeuronType.Recurrent));
                    this.Neurons[i].Initialize(values[i]);
                }
                this.Neurons.Add(new Neuron(NeuronType.Output));
            }

        }
        public void CalculateActivations()
        {
            for (int i = 0; i < this.Neurons.Count - 1; i++)
            {
                Hidden_states.Add( Neurons[i].CalculateActivation() + SumUntilIndex(i) );
            }
            this.Neurons.Last().Initialize(Hidden_states.Last());
            this.Neurons.Last().CalculateActivation();

            Predicted = Neurons.Last().Activation;

            Error();
            ErrorDerivative();
        }
        public static double TanhDerivative(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }
        public void Backpropagate()
        {
            List<double> derivative_states = new List<double>();
            List<double> derivative_states_for_bias = new List<double>();
            for (int i = Neurons.Count - 1; i >= 0; i--)
            {
                derivative_states.Add(Neurons[i].CalculateDerivative(errorDerivative));
                derivative_states_for_bias.Add(Neurons[i].CalculateDerivative(errorDerivative));
                double weight = ProductUntilIndex(derivative_states);
                double bias = ProductUntilIndex(derivative_states_for_bias);

                Neurons[i].UpdateWeights(weight);
                Neurons[i].UpdateBias(bias);
            }
        }
        public double Error()
        {
            this.error = 0.5 * Math.Pow(Predicted - Actual, 2);
            return error;
        }
        public double ErrorDerivative()
        {
            errorDerivative = Predicted - Actual;
            return errorDerivative;
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
        private double ProductUntilIndex(List<double> list/*, int index*/)
        {
            double sum = 0;
            int localIndex = 0;
            while (localIndex < list.Count)
            {
                sum *= list[localIndex];
                localIndex++;
            }
            return sum;
        }
    }
}
