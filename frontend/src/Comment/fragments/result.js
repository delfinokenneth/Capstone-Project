import '../comment.css';

const displayResult =(props) =>
{
    return(
        <div className="resultBox">
            <h4>RESULT</h4>
            <label><b>{props.sentiment}</b></label>
        </div>
    );  
}

export default displayResult;