package bit.lin.pairwise.ranknet;

import java.io.File;
import java.util.Vector;

import org.joone.engine.FullSynapse;
import org.joone.engine.InputPatternListener;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.Pattern;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.net.NeuralNet;

import bit.lin.nn.CustomizedNN;
import bit.lin.utils.JUtils;

public class RankNet extends CustomizedNN {
	String _train = "/home/lins/data/Learning to rank/10k4j/Fold1/train.txt";

	private RankNet() {
		_nn = new NeuralNet4RN();
	}

	public static RankNet restoreNeuralNet(String fName) {
		RankNet rn = new RankNet();
		JUtils ju = new JUtils();
		rn.setNeuralnet(ju.restoreNeuralNet(fName));
		System.out.println("Neural Net has been deserialized!");
		return rn;
	}

	public RankNet(int hiddenCount) {
		_nn = new NeuralNet4RN();
		// initial layers
		SigmoidLayer input = new SigmoidLayer("layer1 input");
		SigmoidLayer hidden = new SigmoidLayer("layer2 hidden");
		SigmoidLayer output = new SigmoidLayer("layer3 output");
		input.setRows(136);
		hidden.setRows(hiddenCount);
		output.setRows(1);
		
		// initial synapses
		FullSynapse fs4i2h = new FullSynapse();
		FullSynapse fs4h2o = new FullSynapse();
		TeachingSynapse ts = new TeachingSynapse();
		ts.setTheTeacherSynapse(new TeacherSynapse4RN());
		
		// link synapses to layers
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
	}
	public static void main(String[] args){
		RankNet rn = new RankNet(10);
	}
}
