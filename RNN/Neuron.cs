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
        public double Input { get; set; }
        public double Bias { get; set; }
        public double Weight { get; set; }
        public Neuron()
        {
            Activation = 0;
            Bias = rnd.NextDouble();
            Weight = rnd.NextDouble();
        }
        public void Initialize(double input)
        {
            Input = input;
        }
        public double CalculateActivation()
        {
            Activation = Math.Tanh(Weight * Input + Bias);
            return Activation;
        }
    }
}
