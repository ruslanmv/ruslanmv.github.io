let model;
let imagenetResult;
let predictDataKey = "beagle";
let selectedDiv = undefined;

let dataLookup = {

	beagle: {

		relativeDiv: "data1",
		dataUrl: "../../assets/data/beagle.json",
		imageUrl: "../../assets/img/playground/beagle.jpg"

	},

	macaw: {

		relativeDiv: "data2",
		dataUrl: "../../assets/data/macaw.json",
		imageUrl: "../../assets/img/playground/macaw.jpg"

	},

	tigerShark: {

		relativeDiv: "data3",
		dataUrl: "../../assets/data/tigerShark.json",
		imageUrl: "../../assets/img/playground/tigerShark.jpg"

	},

	golfBall: {

		relativeDiv: "data4",
		dataUrl: "../../assets/data/golfBall.json",
		imageUrl: "../../assets/img/playground/golfBall.jpg"

	},

	golfCart: {

		relativeDiv: "data5",
		dataUrl: "../../assets/data/golfCart.json",
		imageUrl: "../../assets/img/playground/golfCart.jpg"

	}

};

$(function() {

	$.ajax({
		url: '../../assets/data/imagenet_result.json',
		type: 'GET',
		async: true,
		dataType: 'json',
		success: function (data) {

			imagenetResult = data;
			createModel();

		}
	});

	$("#selector > main > div > img").click(function() {
		$(this).css("border", "1px solid #6597AF");
		if (selectedDiv !== undefined) {
			$("#" + selectedDiv).css("border", "0");
		}
		selectedDiv = $(this).attr('id');
	});

	$("#cancelPredict").click(function() {
		hideSelector()
	});

	$("#selectorCurtain").click(function() {
		hideSelector();
	});

	$("#selectorTrigger").click(function() {
		showSelector();
	});

	$("#executePredict").click(function() {

		updatePredictDataKey();
		hideSelector();
		getDataAndPredict(function(finalResult) {
			$("#labelImage").attr("src", dataLookup[ predictDataKey ].imageUrl);
			console.log(generateInference( finalResult ));
		});

	});

});

function createModel() {

	let container = document.getElementById("modelArea");

	model = new TSP.models.Sequential( container, {

		stats: true

	} );

	model.add( new TSP.layers.RGBInput() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Dense( {
		
		paging: true,
		segmentLength: 400

	} ) );

	model.add( new TSP.layers.Dense( {
		
		paging: true,
		segmentLength: 400

	} ) );

	model.add( new TSP.layers.Output1d( {
		
		paging: true,
		segmentLength: 400,
		outputs: imagenetResult

	} ) );

	model.load( {

		type: "tensorflow",
		url: "../../assets/model/alexnet/model.json",
		outputsName: [ "norm1", "pool1", "norm2", "pool2", "conv3_1", "conv4_1", "conv5_1", "pool5", "Relu", "Relu_1", "Softmax" ],
		
		onProgress: function( fraction ) {
			
			$("#downloadProgress").html( ( 100 * fraction ).toFixed( 2 ) + "%" );
			
		},
		
		onComplete: function() {
			
			$("#downloadNotice").hide();
			$("#creationNotice").show();
			
		}

	} );

	model.init( function() {

		getDataAndPredict( function( finalResult ) {
			$( "#loadingPad" ).hide();

			generateInference( finalResult );

		} );

	} );

}

function getDataAndPredict( callback ) {

	$.ajax({
		url: dataLookup[ predictDataKey ].dataUrl,
		type: 'GET',
		async: true,
		dataType: 'json',
		success: function (data) {

			model.predict( data, function( finalResult ){

				if ( callback !== undefined ) {
					callback( finalResult );
				}

			} );

		}
	});

}

function showSelector() {
	$("#selector").show();
	$("#selectorCurtain").show();
}

function hideSelector() {
	$("#selector").hide();
	$("#selectorCurtain").hide();
	if (selectedDiv !== undefined) {
		$("#" + selectedDiv).css("border", "0");
	}
	selectedDiv = undefined;
}

function updatePredictDataKey() {

	for ( let key in dataLookup ) {

		if ( dataLookup[ key ].relativeDiv === selectedDiv ) {

			predictDataKey = key;
			break;

		}

	}

}

function generateInference( finalResult ) {

	let maxIndex = 0;

	for ( let i = 1; i < finalResult.length; i ++ ) {

		maxIndex = finalResult[ i ] > finalResult[ maxIndex ] ? i : maxIndex;

	}

	$("#PredictResult").text(imagenetResult[ maxIndex ]);

}