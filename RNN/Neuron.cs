using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNN
{
    internal class Neuron
    {
        static Random rnd = new Random();
        public double Activation { get; set; }
        public NeuronType Type { get; set; }
        public double Input { get; set; }
        public double Bias { get; set; }
        public double Weight { get; set; }
        public double Derivative { get; set; }

        #region Contructor
        public Neuron()
        {
            Activation = 0;
            Bias = rnd.NextDouble();
            Weight = rnd.NextDouble();
        }
        public Neuron(NeuronType type) 
            : this()
        {
            this.Type = type;
        }
        #endregion // Constructor

        public void Initialize(double input)
        {
            Input = input;
        }
        public double CalculateActivation()
        {
            if (this.Type == NeuronType.Recurrent)
            {
                Activation = Math.Tanh(Weight * Input + Bias);
            }
            else
            {
                Activation = Weight * Input + Bias;
            }
            
            return Activation;
        }
        public void UpdateWeights(double errorDerivative)
        {
            this.Weight = Weight - RecurrentNeuralNetwork.LearningRate * errorDerivative;
        }
        public void UpdateBias(double errorDerivative)
        {
            this.Bias = Bias - RecurrentNeuralNetwork.LearningRate * errorDerivative;
        }
        public double CalculateDerivative(double errorDerivative)
        {
            if (this.Type == NeuronType.Recurrent)
            {
                Derivative = RecurrentNeuralNetwork.TanhDerivative(this.Activation) * errorDerivative * Input;
            }
            else
            {
                Derivative = errorDerivative * Input;
            }
            return Derivative;
        }
    }
}
