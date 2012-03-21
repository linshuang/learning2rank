package bit.lin.sortnet;

import java.io.File;

import org.joone.engine.FullSynapse;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.net.NeuralNet;

import bit.lin.nn.CustomizedNN;
import bit.lin.utils.JUtils;

public class SortNet extends CustomizedNN {
	private SortNet() {
		_nn = new NeuralNet();
	}

	public SortNet(int hiddenCount) {
		_nn = new NeuralNet();
		// initial layers
		LinearLayer input = new LinearLayer("layer1 input");
		SigmoidLayer hidden = new SigmoidLayer("layer2 hidden");
		SigmoidLayer output = new SigmoidLayer("layer3 output");
		input.setRows(272);
		hidden.setRows(hiddenCount);
		output.setRows(2);
		// initial synapses
		FullSynapse fs4i2h = new FullSynapse();
		FullSynapse fs4h2o = new FullSynapse();
		// input connector
		FilesInputSynapse4SN inputData = new FilesInputSynapse4SN();
		inputData.setBuffered(false);
		inputData.setAdvancedColumnSelector("3-138,141-276");
		inputData
				.setInputFile(
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"),
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"));
		FilesInputSynapse4SN desiredData = new FilesInputSynapse4SN();
		desiredData.setBuffered(false);
		desiredData.setAdvancedColumnSelector("277,278");
		desiredData
				.setInputFile(
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"),
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"));
		// InputConnector inputTraining = new InputConnector();
		// inputTraining.setAdvancedColumnSelector("3-138,141-276");
		// inputTraining.setInputSynapse(inputData);
		// inputTraining.setBuffered(false);
		// InputConnector inputValidating = new InputConnector();
		// inputValidating.setAdvancedColumnSelector("277,278");
		// inputValidating.setInputSynapse(inputData);
		// inputValidating.setBuffered(false);
		// teaching synapse
		TeachingSynapse ts = new TeachingSynapse();
		// ts.setDesired(inputValidating);
		ts.setDesired(desiredData);
		// link synapses to layers
		// input.addInputSynapse(inputTraining);
		input.addInputSynapse(inputData);
		input.addOutputSynapse(fs4i2h);
		hidden.addInputSynapse(fs4i2h);
		hidden.addOutputSynapse(fs4h2o);
		output.addInputSynapse(fs4h2o);
		output.addOutputSynapse(ts);
		// add synapses and layers to neural net
		_nn.addLayer(input, NeuralNet.INPUT_LAYER);
		_nn.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		_nn.addLayer(output, NeuralNet.OUTPUT_LAYER);
		_nn.setTeacher(ts);
		// configure
		Monitor m = _nn.getMonitor();
		m.setLearningRate(super._learningRate);
		m.setMomentum(_momentum);
		m.setTrainingPatterns(_patterns);
		m.setTotCicles(_cicles);
		m.setLearning(true);
		m.addNeuralNetListener(this);
		m.setBatchSize(1);
		m.addLearner(0, "bit.lin.sortnet.SortNetLearner");
		m.setLearningMode(0);

	}

	public static SortNet restoreNeuralNet(String fName) {
		SortNet sn = new SortNet();
		JUtils ju = new JUtils();
		sn.setNeuralnet(ju.restoreNeuralNet(fName));
		System.out.println("Neural Net has been deserialized!");
		return sn;
	}

	public static void main(String[] args) {
//		SortNet sn = new SortNet(10);
//		sn.train(true);
		System.out.println(System.getenv().get("PATH"));
	}
}
