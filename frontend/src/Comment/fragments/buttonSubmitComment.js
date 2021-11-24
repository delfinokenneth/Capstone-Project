import { Button } from 'react-bootstrap';
import '../comment.css';
const submitComment = (props) =>
{
    return (
        <div className="btnSubmitFragment">
            <Button 
            className="btn btn-success" 
            id="btnSubmit"
            onClick={props.submitComment}>Submit</Button>
        </div>
    );
}

export default submitComment;
