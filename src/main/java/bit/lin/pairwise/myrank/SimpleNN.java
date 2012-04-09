package bit.lin.pairwise.myrank;

import java.io.Serializable;
import java.util.Vector;

import org.joone.engine.FullSynapse;
import org.joone.engine.Layer;
import org.joone.engine.Monitor;
import org.joone.engine.Pattern;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

import bit.lin.utils.JUtils;

public class SimpleNN implements SubClassifier, Serializable {
	NeuralNet _nn;
	private static final long serialVersionUID = 2237205741957802819L;

	PrfIntervalClassifier _pic;

	String _name;
	int[] _pId = new int[2];
	transient String _doc1 = null;
	transient String _doc2 = null;
	boolean isPatternSuitable = false;
	int _step0 = 0;
	int _step1 = 0;
	double _learningRate;

	public SimpleNN(String name, PrfIntervalClassifier pic) {
		_name = name;
		_pic = pic;
		construct();
	}

	public void setInputPatterns4t(String[] pair) {
		// 等待上一次的训练结束
		int count = 1;
		while (_nn.isRunning()) {
			try {
				Thread.sleep(count * 500);
				if (count < 5)
					count++;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		isPatternSuitable = false;
		_doc1 = pair[0];
		_doc2 = pair[1];
		int interval = (new Integer(pair[0].split(";")[0]))
				- (new Integer(pair[1].split(";")[0]));
		if (contains(interval)) {
			isPatternSuitable = true;
			addInput(new String[] { _doc1, _doc2 });
			addOutput4t(interval);
			if (interval == _pId[0]) {
				_step0++;
				_learningRate = 1.0 / (1.0 + _step0);
			} else {
				_step1++;
				_learningRate = 1.0 / (1.0 + _step1);
			}
		}
	}

	transient JUtils _jUtils;

	private void addInput(String[] strs) {
		if (_jUtils == null)
			_jUtils = new JUtils();
		_nn.getInputLayer().removeAllInputs();
		MemoryInputSynapse mis4m2i = new MemoryInputSynapse();
		mis4m2i.setAdvancedColumnSelector("3-138,140-275");
		mis4m2i.setInputArray(_jUtils.Str2Doubles(strs));
		_nn.getInputLayer().addInputSynapse(mis4m2i);
	}

	private void addOutput4t(int t) {
		_nn.getTeacher().resetInput();
		MemoryInputSynapse mis4ts = new MemoryInputSynapse();
		mis4ts.setAdvancedColumnSelector("1");
		double[][] tmp = new double[1][1];
		int o;
		if (t == _pId[0])
			o = 0;
		else
			o = 1;

		tmp[0][0] = o;
		mis4ts.setInputArray(tmp);
		_nn.getTeacher().setDesired(mis4ts);
	}

	public void train() {
		if (isPatternSuitable) {
			_nn.getMonitor().setLearningRate(_learningRate);
			_nn.go();
		}
	}

	/**
	 * @return True when preference interval of doc1-doc2 is closer to _pId[0];
	 *         False when preference interval of doc1-doc2 is closer to _pId[1]
	 * @see bit.lin.pairwise.myrank.SubClassifier#classify(java.lang.String,
	 *      java.lang.String)
	 */
	public boolean classify(String doc1, String doc2) {
		addInput(new String[] { doc1, doc2 });
		Layer output = _nn.getOutputLayer();
		output.removeAllOutputs();
		MemoryOutputSynapse menOS = new MemoryOutputSynapse();
		output.addOutputSynapse(menOS);
		_nn.getMonitor().setTotCicles(1);
		_nn.getMonitor().setValidationPatterns(1);
		_nn.getMonitor().setLearning(false);
		_nn.go();
		_nn.stop();
		Vector v = menOS.getAllPatterns();

		Pattern p = (Pattern) v.get(0);
		double[] pattern = p.getArray();

//		System.out.format("score %s - score %s : class %d - class %d : p %f\n",
//				doc1.split(";")[0], doc2.split(";")[0], _pId[0], _pId[1],
//				pattern[0]);
		return pattern[0] < 0.5 ? true : false;
	}

	private boolean contains(int prfInterval) {
		if (_pId[0] == prfInterval || _pId[1] == prfInterval)
			return true;
		return false;
	}

	private void construct() {
		_nn = new NeuralNet();
		// initial layers
		SigmoidLayer input = new SigmoidLayer("layer1 input");
		SigmoidLayer hidden = new SigmoidLayer("layer2 hidden");
		SigmoidLayer output = new SigmoidLayer("layer3 output");
		input.setRows(272);
		hidden.setRows(10);
		output.setRows(1);

		// initial synapses

		FullSynapse fs4i2h = new FullSynapse();
		FullSynapse fs4h2o = new FullSynapse();
		TeachingSynapse ts = new TeachingSynapse();
		MemoryInputSynapse mis4dsd = new MemoryInputSynapse();
		ts.setDesired(mis4dsd);

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
		m.setMomentum(0.2);
		m.setTrainingPatterns(1);
		m.setTotCicles(2);
		m.setLearning(true);
		// m.addNeuralNetListener(this);
		// m.addNeuralNetListener(new DynamicAnnealing());
	}

	public boolean isRunning() {
		return _nn.isRunning();
	}
}
