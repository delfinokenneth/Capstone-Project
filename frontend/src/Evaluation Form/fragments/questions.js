import '../evaluationForm.css';

const displayQuestions = (props) =>
{
    console.log("" + props.sec1[0][0])
    return (
        <div>
            <p>  
                {props.sec1[0][0]}
            </p>
        </div>
    );
}

export default displayQuestions;