using Microsoft.ML;
using System;

namespace DataMining2
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;

        //prediction engine (input and output types)
        public Program()
        {
            //gather any variables and set them
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "model\\model.zip");

            //create a context (connection to the database)
            ctx = new MLContext();

            //create a query to get the data
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');
            //build data pipeline (transforming your data into something that works)

            //train your model (make it run the data)

            //consume your model (use the model to make predictions)

            Console.ReadLine();
        }

        static void Main(string[] args)
        {
            new Program();
        }
    }
}
