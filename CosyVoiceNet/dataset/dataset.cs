// Exported from CosyVoice\cosyvoice\dataset\dataset.py
using CosyVoiceNet.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TorchSharp;
using TorchSharp.Data;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CosyVoiceNet.Dataset
{
    public class Processor : IEnumerable<Tensor>
    {
        private readonly IEnumerable<Tensor> _source;
        private readonly Func<IEnumerable<Tensor>, IEnumerable<Tensor>> _processor;

        public Processor(IEnumerable<Tensor> source, Func<IEnumerable<Tensor>, IEnumerable<Tensor>> processor)
        {
            if (processor == null) throw new ArgumentNullException(nameof(processor));
            _source = source ?? throw new ArgumentNullException(nameof(source));
            _processor = processor;
        }

        public void SetEpoch(int epoch)
        {
            if (_source is DistributedSampler sampler)
            {
                sampler.SetEpoch(epoch);
            }
        }

        public IEnumerator<Tensor> GetEnumerator()
        {
            return _processor(_source).GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Processor Apply(Func<IEnumerable<Tensor>, IEnumerable<Tensor>> processor)
        {
            return new Processor(this, processor);
        }
    }

    public class DistributedSampler
    {
        private int _epoch = -1;
        private int _rank;
        private int _worldSize;
        private int _workerId;
        private int _numWorkers;
        private readonly bool _shuffle;
        private readonly bool _partition;

        public DistributedSampler(bool shuffle = true, bool partition = true)
        {
            _shuffle = shuffle;
            _partition = partition;
            Update();
        }

        public void Update()
        {
            _rank = 0; // Default rank
            _worldSize = 1; // Default world size
            _workerId = 0; // Default worker ID
            _numWorkers = 1; // Default number of workers
        }

        public void SetEpoch(int epoch)
        {
            _epoch = epoch;
        }

        public IEnumerable<int> Sample(IEnumerable<int> data)
        {
            var dataList = data.ToList();

            if (_partition)
            {
                if (_shuffle)
                {
                    var random = new Random(_epoch);
                    dataList = dataList.OrderBy(_ => random.Next()).ToList();
                }

                if (dataList.Count < _worldSize)
                {
                    dataList = Enumerable.Repeat(dataList, (int)Math.Ceiling((double)_worldSize / dataList.Count))
                                         .SelectMany(x => x)
                                         .Take(_worldSize)
                                         .ToList();
                }

                dataList = dataList.Where((_, index) => index % _worldSize == _rank).ToList();
            }

            if (dataList.Count < _numWorkers)
            {
                dataList = Enumerable.Repeat(dataList, (int)Math.Ceiling((double)_numWorkers / dataList.Count))
                                     .SelectMany(x => x)
                                     .Take(_numWorkers)
                                     .ToList();
            }

            return dataList.Where((_, index) => index % _numWorkers == _workerId);
        }

        public Dictionary<string, object> SampleInfo()
        {
            return new Dictionary<string, object>
            {
                { "rank", _rank },
                { "world_size", _worldSize },
                { "worker_id", _workerId },
                { "num_workers", _numWorkers }
            };
        }
    }

    public class DataList : IEnumerable<Dictionary<string, object>>
    {
        private readonly List<string> _lists;
        private readonly DistributedSampler _sampler;

        public DataList(IEnumerable<Dictionary<string, object>> data, bool shuffle = true, bool partition = true)
        {
            _lists = data.Select(d => d["src"].ToString()).ToList();
            _sampler = new DistributedSampler(shuffle, partition);
        }

        public DataList(List<string> lists, bool shuffle = true, bool partition = true)
        {
            _lists = lists;
            _sampler = new DistributedSampler(shuffle, partition);
        }

        public void SetEpoch(int epoch)
        {
            _sampler.SetEpoch(epoch);
        }

        public IEnumerator<Dictionary<string, object>> GetEnumerator()
        {
            var samplerInfo = _sampler.SampleInfo(); // Updated to use the new SampleInfo method
            var indexes = _sampler.Sample(Enumerable.Range(0, _lists.Count));

            foreach (var index in indexes)
            {
                var data = new Dictionary<string, object>
                {
                    { "src", _lists[index] }
                };
                foreach (var kv in samplerInfo)
                {
                    data[kv.Key] = kv.Value;
                }
                yield return data;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    // Adjust DatasetFactory to resolve type mismatches
    public static class DatasetFactory
    {
        public static IEnumerable<Dictionary<string, object>> Dataset(string dataListFile,
                                        List<Func<IEnumerable<Dictionary<string, object>>, IEnumerable<Dictionary<string, object>>>> dataPipeline,
                                        string mode = "train",
                                        bool gan = false,
                                        bool dpo = false,
                                        bool shuffle = true,
                                        bool partition = true)
        {
            var lists = FileUtils.ReadLists(dataListFile);
            var dataset = new DataList(lists, shuffle, partition);
    
            foreach (var func in dataPipeline)
            {
                dataset = new DataList(func(dataset), shuffle, partition);
            }

            return dataset;
        }
    }
}