using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining2
{
    public class DisneylandPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint Label { get; set; }
        public float[] Score { get; set; }
        public float[] Probability { get; set; }
    }
}
