import '../evaluationForm.css';

const displayQuestions = (props) =>
{
    const questions = Object.values(props.sec1[0])
    console.log(questions)
    return (
        <div className="questionaireDiv">
            <table className="questionaireTable">
                <thead>
                    <tr>
                        <th>
                            comment
                        </th>
                    </tr>
                </thead>
                <tbody>
                {
                questions.map((comment) => {
                        return (
                            <tr>
                                <td>
                                    {comment}
                                </td>
                            </tr>
                            )
                })}
                </tbody>
            </table>
        </div>
    );
}

export default displayQuestions;