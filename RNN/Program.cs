namespace RNN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // need some sort of loop
            // only 1 neuron
            // First neuron = tanh or sigmoid(weight * first input + bias) = h1
            // second neuron = h1 + tanh or sigmoid(weight * second input + bias) = h2
            // third neuron = h1 + h2 + tanh or sigmoid(weight * third input + bias) = h3
            // output = sigmoid(h1 + h2 + h3) 
            // from here the same thing as before but only one weight and one bias (???)

            // cut data into 4+1 where 4 is the input and 1 is the value the model has to predict

            List<string> inputData = File.ReadAllLines("amzn.csv").Skip(1).Select(x => x).ToList();
            List<string> helper = new List<string>();   
            
            RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork();

            int index = 0;
            for (int i = 0; i < inputData.Count; i++)
            {
                if (index < 5)
                {
                    helper.Add(inputData[i]);
                    index++;
                }
                else if(index == 5)
                {
                    // training and testing loop
                    rnn.Initialize(helper);
                    rnn.CalculateActivations();
                    Console.WriteLine($"Error: {rnn.Error()}");
                    helper = new List<string>();
                    index = 0;
                }
            }

        }
    }
}
